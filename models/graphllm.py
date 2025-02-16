import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from models.gnns import load_gnn_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math
from torch_geometric.data import Data  # ✅ Fixes the NameError



class GraphLLM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            # "max_memory": {0: '30GiB', 1: '30GiB'},
            "max_memory": {0: '50GiB'},
            # "max_memory": {0: '7GiB', 1: '7GiB', 2: '7GiB', 3: '7GiB', 4: '7GiB', 5: '7GiB', 6: '7GiB', 7: '7GiB'},
            # "max_memory": {0: '0GiB', 1: '20GiB', 2: '20GiB', 4: '20GiB'},
            # "max_memory": {0: '0GiB', 1: '30GiB', 2: '10GiB', 3: '10GiB', 4: '10GiB'},
            # "max_memory": {0: '0GiB', 1: '10GiB', 2: '10GiB', 3: '20GiB', 4: '20GiB'},
            # "max_memory": {0: '30GiB', 1: '30GiB', 2: '30GiB', 3: '30GiB'},
            # "max_memory": {1: '40GiB', 2: '40GiB', 3: '40GiB', 4: '40GiB', 5: '40GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        
        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get special tokens from tokenizer
        self.eos_token = self.tokenizer.eos_token
        
        # Construct chat format tokens
        self.BOS = '<s>[INST]'
        self.EOS_USER = '[/INST]'
        self.EOS = '</s>'
        self.IGNORE_INDEX = -100

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )

        print("Haan ji")
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.llm_model_path,
        #     torch_dtype=torch.float16,
        #     device_map={"": "cuda:0"},  # ✅ Force entire model onto GPU
        #     low_cpu_mem_usage=True
        # )


        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for param in model.parameters():
                param.requires_grad = False
        else:
            if args.finetune_method == 'full':
                print("Full-parameter finetuning of LLAMA!")
                model.gradient_checkpointing_enable()
                for param in model.parameters():
                    param.requires_grad = True
            elif args.finetune_method == 'lora':
                print("Training LLAMA with LORA!")
                model = prepare_model_for_kbit_training(model)
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        # Graph encoder setup
        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.get_input_embeddings()
        
        self.no_graph_embedding = nn.Parameter(
        torch.randn(1, 1, args.gnn_hidden_dim) / math.sqrt(args.gnn_hidden_dim)
        )
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=args.gnn_hidden_dim,
            num_heads=4,
            batch_first=True
        )

    def encode_graphs(self, graphs_list):
        """
        Encode graphs for the planner, handling empty graph lists
        """
        if not graphs_list:
            # Return zero tensor if no graphs
            return torch.zeros((1, 1, self.projector[0].in_features), device=self.model.device)
            
        graph_embeds = []
        for graph in graphs_list:
            try:
                graph = graph.to(self.model.device)
                n_embeds, _ = self.graph_encoder(graph.x, graph.edge_index.long(), graph.edge_attr)
                # Mean pooling for each graph
                g_embed = scatter(n_embeds, 
                                torch.zeros(n_embeds.size(0), dtype=torch.long, device=self.model.device),
                                dim=0,
                                reduce='mean')
                graph_embeds.append(g_embed)
            except (AttributeError, ValueError) as e:
                print(f"Warning: Error processing graph: {e}")
                continue
        
        if not graph_embeds:  # If all graphs failed processing
            return torch.zeros((1, 1, self.projector[0].in_features), device=self.model.device)
        
        # Stack and mean pool across graphs
        g_embeds = torch.stack(graph_embeds).mean(dim=0)  # [1, hidden_dim]
        return g_embeds.unsqueeze(0)  # [1, 1, hidden_dim]
    
    def forward(self, samples):
        # Tokenize inputs and labels

        flattened_inputs = [" ".join(input_list) if isinstance(input_list, list) else str(input_list) for input_list in samples['input']]
        # flattened_labels = [" ".join(input_list) if isinstance(input_list, list) else str(input_list) for input_list in samples['label']]


        inputs = self.tokenizer(flattened_inputs, add_special_tokens=False)
        # labels = self.tokenizer(flattened_labels, add_special_tokens=False)

        # inputs = self.tokenizer(samples['input'], add_special_tokens=False)
        labels = self.tokenizer(samples['label'], add_special_tokens=False)

        # Get special token ids
        eos_tokens = self.tokenizer(self.EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
        ).to(self.model.device)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).to(self.model.device)

        batch_size = len(samples['input'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        # First pass - get all embeddings and lengths
        for i in range(batch_size):
            # Encode graphs
            
            graph_embeds = self.encode_graphs(samples['graphs'][i])  # [1, 1, hidden_dim]
            assert graph_embeds.size(-1) == self.projector[0].in_features, \
                f"Graph embedding dimension mismatch: {graph_embeds.size(-1)} vs {self.projector[0].in_features}"
            graph_embeds = self.projector(graph_embeds.squeeze(1))  # [1, proj_dim]

            # Prepare label sequence first (following G-Retriever)
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids

            # Now include label in input sequence (G-Retriever style)
            input_ids = (inputs.input_ids[i][:self.max_txt_len] + 
                        eos_user_tokens.input_ids + 
                        label_input_ids)  # Include labels in input

            # Create embeddings
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([
                bos_embeds,
                graph_embeds,
                inputs_embeds
            ], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [self.IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # Get maximum length
        max_length = max([x.shape[0] for x in batch_inputs_embeds])

        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [self.IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        # Stack all tensors
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        # Forward pass with autocast
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):
        # encode inputs
        # print("type of samples:", type(samples['input']))
        # # print()
        # print(samples['input'])

        print("model device:", self.model.device)

        # Input validation function
        def validate_tensor(tensor, name):
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN values detected in {name}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Inf values detected in {name}")

        # Ensure each batch sample is a string by joining inner lists
        flattened_inputs = [" ".join(input_list) if isinstance(input_list, list) else str(input_list) for input_list in samples['input']]

        # print("type of samples:", type(flattened_inputs))
        # # print()
        # print(flattened_inputs)

        # Now call the tokenizer safely
        inputs = self.tokenizer(flattened_inputs, add_special_tokens=False)

        # inputs = self.tokenizer(samples['input'], add_special_tokens=False)
        
        # encode special tokens
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device)
        ).to(self.model.device)
        # bos_embeds = self.word_embedding(
        #     self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids.to(self.model.device, dtype=torch.long)
        # )
        # validate_tensor(bos_embeds, "bos_embeds")

        pad_token_tensor = torch.tensor(self.tokenizer.pad_token_id, device=self.model.device)
        pad_embeds = self.word_embedding(pad_token_tensor).unsqueeze(0)
        # pad_embeds = self.word_embedding(
        #     torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long, device=self.model.device)
        # ).unsqueeze(0)
        # validate_tensor(pad_embeds, "pad_embeds")

        # pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0).to(self.model.device)

        batch_size = len(samples['input'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        for i in range(batch_size):
            # Encode graphs for this sample
            print(samples['graphs'])
            graph_embeds = self.encode_graphs(samples['graphs']).to(self.model.device)  # [1, 1, hidden_dim]
            # validate_tensor(graph_embeds, f"graph_embeds batch {i}")

            graph_embeds = self.projector(graph_embeds.squeeze(1))  # [1, proj_dim]
            # validate_tensor(graph_embeds, f"projected_graph_embeds batch {i}")
            
            
            # Add special tokens and create input embeddings
            input_ids = inputs.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self.model.device))

            # input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.model.device)
            # inputs_embeds = self.word_embedding(input_ids_tensor)

            # validate_tensor(inputs_embeds, f"inputs_embeds batch {i}")
            
            # Concatenate all embeddings
            inputs_embeds = torch.cat([
                bos_embeds,
                graph_embeds,
                inputs_embeds
            ], dim=0)
            # validate_tensor(inputs_embeds, f"concatenated_embeds batch {i}")

            # batch_inputs_embeds.append(inputs_embeds)
            batch_inputs_embeds.append(inputs_embeds.to(self.model.device))

            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # Pad inputs to max length
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            if pad_length > 0:
                batch_inputs_embeds[i] = torch.cat([
                    pad_embeds.repeat(pad_length, 1).to(self.model.device), 
                    batch_inputs_embeds[i].to(self.model.device)
                ])
                batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        # Stack tensors
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask,  device=self.model.device)
        # attention_mask = (attention_mask > 0).float().to(self.model.device)  # Ensure it's float


        # validate_tensor(inputs_embeds, "final_inputs_embeds")

        # Print shapes and stats for debugging
        # print(f"inputs_embeds shape: {inputs_embeds.shape}")
        # print(f"attention_mask shape: {attention_mask.shape}")
        # print(f"inputs_embeds range: [{inputs_embeds.min():.3f}, {inputs_embeds.max():.3f}]")

        # print(f"attention_mask values: {attention_mask.sum(dim=1)}")  # Should show number of non-pad tokens per batch

        # inputs_embeds = torch.clamp(inputs_embeds, min=0, max=1)
        # print(f"inputs_embeds range after clamping: [{inputs_embeds.min():.3f}, {inputs_embeds.max():.3f}]")
        # Before calling model.generate(), check for NaNs and invalid values
        # validate_tensor(inputs_embeds, "final_inputs_embeds before generate")
        # validate_tensor(attention_mask, "attention_mask before generate")

        # if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
        #     raise ValueError("NaN or Inf detected in inputs_embeds before generate!")

        # if (inputs_embeds < 0).any():
        #     print(f"Warning: Negative values found in inputs_embeds!")
                  
       

        # Generate with autocast
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True, 
                temperature=0.4,  # Avoid extremely sharp probabilities
                top_p=0.8,
                top_k=50
                # do_sample=False,     # Disable sampling completely
                # num_beams=1,        # Use greedy search
                # pad_token_id=self.tokenizer.pad_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
                # repetition_penalty=1.0,
                # length_penalty=1.0,
                # no_repeat_ngram_size=0,
                # early_stopping=True
                # pad_token_id=self.tokenizer.pad_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
            )
            # outputs = self.model.generate(
            #     inputs_embeds=inputs_embeds,
            #     max_new_tokens=self.max_new_tokens,
            #     attention_mask=attention_mask,
            #     use_cache=True,
            #     # Add these parameters:
            #     temperature=0.7,  # Lower temperature for more focused sampling
            #     top_p=0.9,       # Nucleus sampling parameter
            #     top_k=50,        # Limit vocabulary to top K tokens
            #     num_beams=3,     # Use beam search
            #     do_sample=True,  # Enable sampling
            # )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        return {
            'input': samples['input'],
            'pred': predictions,
            'label': samples['label'],
        }
        
    # def inference(self, samples):
    #     print("model device:", self.model.device)  # Debugging line to check device

    #     device = self.model.device  # Get model's device (cuda:0 or cpu)

    #     # Ensure each batch sample is a string by joining inner lists
    #     flattened_inputs = [" ".join(input_list) if isinstance(input_list, list) else str(input_list) for input_list in samples['input']]

    #     # Tokenize inputs
    #     inputs = self.tokenizer(flattened_inputs, add_special_tokens=False)

    #     # Move special tokens to the correct device
    #     eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
    #     # bos_embeds = self.word_embedding(
    #     #     self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
    #     # ).to(device)
    #     # pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device=device)).unsqueeze(0)

    #     # Ensure input tensors are moved to the correct device before embedding
    #     bos_ids = self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids.to(device)  # ✅ Move to GPU
    #     bos_embeds = self.word_embedding(bos_ids).squeeze(0)  # ✅ Ensure correct shape and on cuda:0

    #     pad_ids = torch.tensor([self.tokenizer.pad_token_id], device=device)  # ✅ Ensure pad token is on cuda:0
    #     pad_embeds = self.word_embedding(pad_ids).unsqueeze(0)  # ✅ Correct shape


    #     batch_size = len(samples['input'])
    #     batch_inputs_embeds = []
    #     batch_attention_mask = []

    #     for i in range(batch_size):
    #         # Encode graphs for this sample and ensure they are on the same device
    #         # graph_embeds = self.encode_graphs(samples['graphs'][i]).to(device)  # ✅ Move graph embeddings to correct device
    #         if not samples['graphs']:  # ✅ Check if the graphs list is empty
    #             print(f"Warning: Sample {i} has no valid graphs.")
    #             graph_embeds = torch.zeros((1, 1, self.projector[0].in_features), device=device)  # Return zero tensor if no graphs
    #         else:
    #             # Ensure we have a list of graphs and each graph is moved to the correct device
    #             graph_list = samples['graphs']
    #             if isinstance(graph_list, Data):  # ✅ If only one graph, wrap it in a list
    #                 graph_list = [graph_list]

    #             # Encode all graphs and move to device
    #             graph_embeds = self.encode_graphs(graph_list).to(device)


    #         graph_embeds = self.projector(graph_embeds.squeeze(1)).to(device)  # ✅ Ensure projected embeddings are on the correct device

    #         # Convert tokenized input to tensor and move to the correct device
    #         input_ids = torch.tensor(inputs.input_ids[i][:self.max_txt_len], device=device)  # ✅ Ensure input IDs are on device
    #         eos_user_ids = torch.tensor(eos_user_tokens.input_ids, device=device)  # ✅ Ensure eos_user tokens are on device

    #         # Create input embeddings
    #         inputs_embeds = self.word_embedding(input_ids)
            
    #         # Concatenate all embeddings
    #         inputs_embeds = torch.cat([bos_embeds, graph_embeds, inputs_embeds], dim=0)

    #         batch_inputs_embeds.append(inputs_embeds)
    #         batch_attention_mask.append([1] * inputs_embeds.shape[0])

    #     # Pad inputs to max length
    #     max_length = max([x.shape[0] for x in batch_inputs_embeds])
    #     for i in range(batch_size):
    #         pad_length = max_length - batch_inputs_embeds[i].shape[0]
    #         if pad_length > 0:
    #             batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
    #             batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

    #     # Stack tensors and move them to the correct device
    #     inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)  # ✅ Ensure input embeddings are on device
    #     attention_mask = torch.tensor(batch_attention_mask, device=device)  # ✅ Ensure attention mask is on device

    #     # Generate with autocast
    #     with self.maybe_autocast():
    #         outputs = self.model.generate(
    #             inputs_embeds=inputs_embeds,
    #             max_new_tokens=self.max_new_tokens,
    #             attention_mask=attention_mask,
    #             use_cache=True, 
    #         )

    #     predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    #     return {
    #         'input': samples['input'],
    #         'pred': predictions,
    #         'label': samples['label'],
    #     }



    # def inference(self, samples):
    #     print("model device:", self.model.device)  # Debugging line to check device

    #     device = self.model.device  # Get model's device (cuda:0 or cpu)

    #     # Ensure each batch sample is a string by joining inner lists
    #     flattened_inputs = [" ".join(input_list) if isinstance(input_list, list) else str(input_list) for input_list in samples['input']]

    #     print("Flattened inputs:", flattened_inputs)
    #     # Tokenize inputs
    #     inputs = self.tokenizer(flattened_inputs, add_special_tokens=False)

    #     # Move special tokens to the correct device
    #     eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)

    #     # Ensure input tensors are moved to the correct device before embedding
    #     bos_ids = self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids.to(device)  # ✅ Move to GPU
    #     bos_embeds = self.word_embedding(bos_ids).squeeze(0)  # ✅ Ensure correct shape and on cuda:0

    #     pad_ids = torch.tensor([self.tokenizer.pad_token_id], device=device)  # ✅ Ensure pad token is on cuda:0
    #     pad_embeds = self.word_embedding(pad_ids).squeeze(0)  # ✅ Ensure correct shape [embedding_dim]

    #     batch_size = len(samples['input'])
    #     batch_inputs_embeds = []
    #     batch_attention_mask = []

    #     print("Batch size: ", batch_size)
    #     for i in range(batch_size):
    #         print(samples['graphs'])
    #         # Handle graph encoding
    #         if not samples['graphs']:  # ✅ Check if the graphs list is empty
    #             print(f"Warning: Sample {i} has no valid graphs.")
    #             graph_embeds = torch.zeros((1, self.projector[0].in_features), device=device)  # ✅ Return zero tensor if no graphs
    #         else:
    #             # Ensure we have a list of graphs and each graph is moved to the correct device
    #             graph_list = samples['graphs']
    #             if isinstance(graph_list, Data):  # ✅ If only one graph, wrap it in a list
    #                 graph_list = [graph_list]
                
    #             print("Graph list: ", graph_list)
    #             # Encode all graphs and move to device
    #             graph_embeds = self.encode_graphs(graph_list).to(device)

    #         graph_embeds = self.projector(graph_embeds.squeeze(0)).to(device)  # ✅ Ensure projected embeddings are on the correct device

    #         # Convert tokenized input to tensor and move to the correct device
    #         input_ids = torch.tensor(inputs.input_ids[i][:self.max_txt_len], device=device)  # ✅ Ensure input IDs are on device
    #         eos_user_ids = torch.tensor(eos_user_tokens.input_ids, device=device)  # ✅ Ensure eos_user tokens are on device

    #         # Create input embeddings
    #         inputs_embeds = self.word_embedding(input_ids)

    #         # Concatenate all embeddings
    #         inputs_embeds = torch.cat([bos_embeds, graph_embeds, inputs_embeds], dim=0)

    #         batch_inputs_embeds.append(inputs_embeds)
    #         batch_attention_mask.append([1] * inputs_embeds.shape[0])

    #     # Pad inputs to max length
    #     max_length = max(x.shape[0] for x in batch_inputs_embeds)
    #     for i in range(batch_size):
    #         pad_length = max_length - batch_inputs_embeds[i].shape[0]
    #         if pad_length > 0:
    #             repeated_pad = pad_embeds.unsqueeze(0).expand(pad_length, -1)  # ✅ Correctly shape padding
    #             batch_inputs_embeds[i] = torch.cat([repeated_pad, batch_inputs_embeds[i]], dim=0)
    #             batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

    #     # Stack tensors and move them to the correct device
    #     inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)  # ✅ Ensure input embeddings are on device
    #     attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)  # ✅ Ensure attention mask is on device

    #     # Generate with autocast
    #     with self.maybe_autocast():
    #         outputs = self.model.generate(
    #             inputs_embeds=inputs_embeds,
    #             max_new_tokens=self.max_new_tokens,
    #             attention_mask=attention_mask,
    #             use_cache=True,
    #         )

    #     predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    #     return {
    #         'input': samples['input'],
    #         'pred': predictions,
    #         'label': samples['label'],
    #     }

    def maybe_autocast(self, dtype=torch.bfloat16):
        """Helper for handling autocast"""
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def print_trainable_params(self):
        """Print trainable parameter stats"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f'trainable params: {trainable_params:,d} || '
            f'all params: {all_param:,d} || '
            f'trainable%: {100 * trainable_params / all_param:.2f}%'
        )

    @property
    def device(self):
        return list(self.parameters())[0].device