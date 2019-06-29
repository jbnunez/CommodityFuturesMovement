#hypertesting.py



def corn_hyper(wshape, mshape, xCornWasde_train, yCornWasde_train, 
	xCornWasde_test, yCornWasde_test, xCornMkt_train, yCornMkt_train, 
	xCornMkt_test, yCornMkt_test):

	epochs = 90
	decay = 0.4

	#different dropout ratios to be tests
	dropoutlist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	
	dropout_result_wasde = {}
	for drop in dropoutlist:    
		print("testing dropout: " + str(drop)
		(trainScore, testScore) = test_model(drop, wshape, neurons, epochs, 
			xCornWasde_train, yCornWasde_train, xCornWasde_test, yCornWasde_test)
		dropout_scores_wasde[str(drop)] = testScore
	
	
	dropout_result_mkt = {}
	for drop in dropoutlist:    
		print("testing dropout: " + str(drop)
		(trainScore, testScore) = test_model(drop, mshape, neurons, epochs, 
			xCornMkt_train, yCornMkt_train, xCornMkt_test, yCornMkt_test)
		dropout_scores_mkt[str(drop)] = testScore


	#determine and set best dropout ratio 
	least_error = float("inf")
	best_dropout = None
	keylist = dropout_scores.keys()
	for key in keylist:
		if least_error > dropout_scores[key]:
			best_dropout = key
			least_error = dropout_scores[key]
	d = float(best_dropout)


	#list of lstm sizes
	neuronlist1 = [32, 64, 128, 256, 512]
	#list of non lstm layer sizes
	neuronlist2 = [16, 32, 64]
	#dictionary for results
	

	neurons_result_wasde = {}
	for n1 in neuronlist1:
		neurons = [n1, n1]
		for n2 in neuronlist2:
			#have to get the non lstm layers in the layer lsit
			neurons.append(n2)
			neurons.append(1)
			print("testing lstm size: " + str(neurons))
			trainScore, testScore = (trainScore, testScore) = test_model(drop, 
				wshape, neurons, epochs, xCornWasde_train, yCornWasde_train,
				xCornWasde_test, yCornWasde_test)
			neurons_result_wasde[str(neurons)] = testScore

	neurons_result_mkt = {}
	for n1 in neuronlist1:
		neurons = [n1, n1]
		for n2 in neuronlist2:
			#have to get the non lstm layers in the layer lsit
			neurons.append(n2)
			neurons.append(1)
			print("testing lstm size: " + str(neurons))
			trainScore, testScore = (trainScore, testScore) = test_model(drop, 
				mshape, neurons, epochs, xCornMkt_train, yCornMkt_train, 
				xCornMkt_test, yCornMkt_test)
			neurons_result_mkt[str(neurons)] = testScore


	#find best layers sizes
	least_error = float("inf")
	best_neuron = None
	neuron_keys = neurons_result.keys()
	for key in neuron_keys:
		if least_error > neurons_result[key]:
			best_neuron = key
			least_error = neurons_result[key]
	neurons = [int(s) for s in neurons_result.split(',')]

	return (neurons_result_mkt, neurons_result_wasde, dropout_result_mkt, dropout_result_wasde)




def soy_hyper(wshape, mshape, xSoyWasde_train, ySoyWasde_train, 
	xSoyWasde_test, ySoyWasde_test, xSoyMkt_train, ySoyMkt_train, 
	xSoyMkt_test, ySoyMkt_test):
	epochs = 90
	decay = 0.4
	#different dropout ratios to be tests
	dropoutlist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	
	dropout_result_wasde = {}
	for drop in dropoutlist:    
		print("testing dropout: " + str(drop)
		(trainScore, testScore) = test_model(drop, wshape, neurons, epochs, 
			xSoyWasde_train, ySoyWasde_train, xSoyWasde_test, ySoyWasde_test)
		dropout_scores_wasde[drop] = testScore
	
	dropout_result_mkt = {}
	for drop in dropoutlist:    
		print("testing dropout: " + str(drop)
		(trainScore, testScore) = test_model(drop, mshape, neurons, epochs, 
			xSoyMkt_train, ySoyMkt_train, xSoyMkt_test, ySoyMkt_test)
		dropout_scores_mkt[drop] = testScore


	#determine and set best dropout ratio 
	least_error = float("inf")
	best_dropout = None
	keylist = dropout_scores.keys()
	for key in keylist:
		if least_error > dropout_scores[key]:
			best_dropout = key
			least_error = dropout_scores[key]
	d = float(best_dropout)


	#list of lstm sizes
	neuronlist1 = [32, 64, 128, 256, 512]
	#list of non lstm layer sizes
	neuronlist2 = [16, 32, 64]
	#dictionary for results
	

	neurons_result_wasde = {}
	for n1 in neuronlist1:
		neurons = [n1, n1]
		for n2 in neuronlist2:
			#have to get the non lstm layers in the layer lsit
			neurons.append(n2)
			neurons.append(1)
			print("testing lstm size: " + str(neurons))
			trainScore, testScore = (trainScore, testScore) = test_model(drop, 
				wshape, neurons, epochs, xSoyWasde_train, ySoyWasde_train,
				xSoyWasde_test, ySoyWasde_test)
			neurons_result_wasde[str(neurons)] = testScore

	neurons_result_mkt = {}
	for n1 in neuronlist1:
		neurons = [n1, n1]
		for n2 in neuronlist2:
			#have to get the non lstm layers in the layer lsit
			neurons.append(n2)
			neurons.append(1)
			print("testing lstm size: " + str(neurons))
			trainScore, testScore = (trainScore, testScore) = test_model(drop, 
				mshape, neurons, epochs, xSoyMkt_train, ySoyMkt_train, 
				xSoyMkt_test, ySoyMkt_test)
			neurons_result_mkt[str(neurons)] = testScore


	#find best layers sizes
	least_error = float("inf")
	best_neuron = None
	neuron_keys = neurons_result.keys()
	for key in neuron_keys:
		if least_error > neurons_result[key]:
			best_neuron = key
			least_error = neurons_result[key]
	neurons = [int(s) for s in neurons_result.split(',')]

	return (neurons_result_mkt, neurons_result_wasde, dropout_result_mkt, dropout_result_wasde)
