

Competition profile: https://www.kaggle.com/jordanmicahbennett

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
[Bing Xu's Mxnet Ejection Fraction Irregularity Detection Template](https://github.com/apache/incubator-mxnet/tree/master/example/kaggle-ndsb2)


Adjustment(s)
==== 
	+Removal of classic leNet Architecture seen in get_lenet() in [train.py of old code](https://github.com/apache/incubator-mxnet/tree/master/example/kaggle-ndsb2) 
		+Replaced classic leNet Architecture with Deep Residual Neural Network, as seen in [my modification of train.py](https://github.com/JordanMicahBennett/EJECTION-FRACTION-IRREGULARITY-DETECTION-MODEL/blob/master/Train.py), including get_symbol(), about 60 lines of code including conv_factory; [found elsewhere](https://github.com/freesouls/Deep-Residual-Network-For-MXNet.)
	+Removal of Xavier Initialization.
		+No special initialization of weights was used instead of the Xavier Initialization from original code.

Note
====
Unfortunately, due to the gt 720 GPU used, I could only train up to 20 layers, (as indicated in  n = 2 from get_symbol function in [train.py](https://github.com/JordanMicahBennett/EJECTION-FRACTION-IRREGULARITY-DETECTION-MODEL/blob/master/Train.py))







. Model Synopsis
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
