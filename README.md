Author ~ Jordan Micah Bennett
Aim ~ Kaggle Second Annual Data Science Bowl (global competition)
Outcome ~ 145/755.
Framework ~ Bing Xu's Mxnet Ejection Fraction Irregularity Detection Template.
Adjustment(s) ~ 
	+Doffing of classic leNet Architecture (Substitute ~ Deep Residual Neural Network)
	+Doffing of Xavier Initialization (Substitute ~ None)





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




	
	
	
Public leader-board ranking existed as high as 76/500+, at the boundary of the competition's end:
![Alt text](https://github.com/JordanMicahBennett/EJECTION-FRACTION-IRREGULARITY-DETECTION-MODEL/blob/master/source-code/data/images/captures/0.png?raw=true"default page")