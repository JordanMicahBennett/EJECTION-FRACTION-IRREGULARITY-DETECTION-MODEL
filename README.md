

Overall Competition profile: https://www.kaggle.com/jordanmicahbennett

Contest page: https://www.kaggle.com/c/second-annual-data-science-bowl

Author
====
Jordan Micah Bennett

Aim
====
Kaggle Second Annual Data Science Bowl (global competition)

Outcome
====
145/755.


Original Framework 
====
[Bing Xu's Mxnet Ejection Fraction Irregularity Detection Template](https://github.com/apache/incubator-mxnet/tree/master/example/kaggle-ndsb2).


Crucial Adjustment(s)
==== 
1. Removal of classic leNet Architecture seen in get_lenet() (about 20 lines of code) in [train.py of old code](https://github.com/apache/incubator-mxnet/tree/master/example/kaggle-ndsb2). 

	i. Replaced classic leNet Architecture with Deep Residual Neural Network, as seen in [my modification of train.py](https://github.com/JordanMicahBennett/EJECTION-FRACTION-IRREGULARITY-DETECTION-MODEL/blob/master/Train.py), including get_symbol(), (about 60 lines of code including conv_factory); [found elsewhere](https://github.com/freesouls/Deep-Residual-Network-For-MXNet).
	
2. Removal of Xavier Initialization.

	i. No special initialization of weights was used instead of the Xavier Initialization from original code.

The Winning code, and computation
====
Unfortunately, due to the 60 US dollar gt 720 GPU that I used, I could only train up to **20 layers before my computer actually died after ~20 hours of training**, (as indicated in  n = 2 from get_symbol function in [train.py](https://github.com/JordanMicahBennett/EJECTION-FRACTION-IRREGULARITY-DETECTION-MODEL/blob/master/Train.py))

[The winners Tencia and "Woshialex"](https://github.com/woshialex/diagnose-heart) by comparison, used a gtx 970 GPU and gtx 980 GPU, with a combined cost of almost 2000 US dollars at the time. Their architecture was smart, and crucially, they were also able to use **36 layers** at minimum), based on [their documentation](https://github.com/woshialex/diagnose-heart/blob/master/TenciaWoshialex_model_documentation.pdf), seen in the convolutional neural network architecture, in the table after in section 3. Deep learning models are power hungry beasts :] Faster GPUS allow for both deeper layers, and faster training times, due to more parallelization strategies available in more expensive GPU hardware, the same parallelization needed for matrix multiplication largely found in neural networks.


Model Synopsis
====
	... Input Data Architecture
		*Predominantly, 30 frames per sequence.

	... Input Data Architecture - Processing
		*Condense relevant data amidst multi-channel format vector.
		*Converge on aforesaid multi-channel sequence.

	... Input Data Architecture - Processing ( alternative )
		*Compute difference of channels outcome sigma, over dynamically altering mxnet symbolic interface aligned time series quantization. 

	... Architecture Objective
		*Batch normalization and drop-out aligned convolutional neural network.

	Problem space:
		*Compute binary (_cdf_|0,1), CDF aligned outcome classification on 600 data points, via regression. 


		
		
		
		
		
		
		
. Preprocessing
	*Compose csv aligned data file, of 30 x 64 x 64 tensor line aligned symbols ( via CSVIter par dynamic non-complete data loading )

	
	
	
	
	
	
	

	

. Instruction Cycle
	+Organize data as follows:
	
		-data
		 |
		 ---- sample_submission_validate.csv
		 |
		 ---- train.csv
		 |
		 ---- train
		 |    |
		 |    ---- 0
		 |    |
		 |    ---- …
		 |
		 ---- validate
			  |
			  ---- 501
			  |
			  ---- …
	+Execute preprocessing.py
	+Execute train.py, thus generating relevant details.







	
	
Public leader-board ranking existed as 'high' as 76/500+, at the boundary of the global competition's (kaggle - Second Annual Data Science Bowl) end:
![Alt text](https://github.com/JordanMicahBennett/EJECTION-FRACTION-IRREGULARITY-DETECTION-MODEL/blob/master/data/images/captures/0.png)
