 pretrained_dict  = torch.load(args.resume) 
        best_acc = pretrained_dict['best_acc'] 
        start_epoch = pretrained_dict['epoch'] 
        optimizer_dict = pretrained_dict['optimizer'] 
        pretrained_dict=pretrained_dict['state_dict'] 
        model_dict = model.state_dict() 
                # 1. filter out unnecessary keys 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        # 2. overwrite entries in the existing state dict 
        add_some_raws = torch.zeros(2,32) 
        add_some_raws = add_some_raws.cuda() 
        add_some_elements = torch.zeros(2) 
        add_some_elements = add_some_elements.cuda() 
        List_of_layers_name = [ 
         "module.control_v1.bias", 
         "module.bn_v1.weight", 
         "module.bn_v1.bias", 
         "module.bn_v1.running_mean", 
	"module.bn_v1.running_var"] 
	for name, param in pretrained_dict.items(): 
		if name == "module.control_v1.weight": 
			param = torch.cat((param, add_some_raws), 0) 
			pretrained_dict[name] = param 
		elif name in List_of_layers_name: 
			param = torch.cat((param, add_some_elements), 0) 
			pretrained_dict[name] = param 
			model_dict.update(pretrained_dict) 
		# 3. load the new state dict 
		model.load_state_dict(model_dict) 
		pretrained_dict5  = torch.load('/content/model_best5.pth.tar') 
		optimizer_dict5 = pretrained_dict5['optimizer'] 
		#update param_groups by adding params 0,...,244 
		dict_d1={'params': [0, … , 244]} 
		optimizer_dict["param_groups"][0].update(dict_d1) 
		#update state  #params 
		optimizer_dict["state"][165]["momentum_buffer"] = 
		torch.cat((optimizer_dict["state"][165]["momentum_buffer"], add_some_raws), 0) 
	optimizer_dict["state"][166]["momentum_buffer"] = 
	torch.cat((optimizer_dict["state"][166]["momentum_buffer"], add_some_elements), 0) 
	optimizer_dict["state"][167]["momentum_buffer"] = 
	torch.cat((optimizer_dict["state"][167]["momentum_buffer"], add_some_elements), 0) 
	optimizer_dict["state"][168]["momentum_buffer"] = 
	torch.cat((optimizer_dict["state"][168]["momentum_buffer"], add_some_elements), 0) 
	#keys 
	tmp = 164 
	for i in range(231,241): 
	tmp = tmp + 1 
	optimizer_dict["state"][i] = optimizer_dict["state"].pop(tmp) 
	optimizer_dict5["param_groups"][0].update(optimizer_dict["param_groups"][0]) 
	optimizer_dict5["state"].update(optimizer_dict["state"]) 
	optimizer.load_state_dict(optimizer_dict5)