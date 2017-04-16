Convolutional Neural Network (CNN) for HI-C Interaction Count Prediction

Files		Description
--------------- ---------------------------------------------------------
cnn.py		Usage: python cnn.py <chromosome #> <cell type> <dataset fold>
		Example: python cnn.py 1 Gm12878 0
		Read in trainset data, build & save cnn model, evaluate 
		against testset data, output plots.
load.py		Usage: python cnn.py <chromosome #> <cell type> <dataset fold>
		Example: python load.py 1 Gm12878 0
		Load existing cnn model, evaluate against testset data, output
		plots.
plots.py	Contains functions to output plots: Pearson correlation coefficient
		by pairwise distance, histogram of interaction counts by pairwise
		distance, mean/standard deviation for actual counts vs predicted
		counts by pairwise distance, scatterplot of predicted vs actual
		interaction counts.
input/<choromosome #>/sequence.txt
		<region index> \t <k-mer signal 1> \t <k-mer signal 2> \t ... \n
input/<choromosome #>/<cell type>/markers.txt
		<region index> \t <chromatin marker signal 1> \t <signal 2> \t ... \n
input/<choromosome #>/<cell type>/train<dataset fold>.txt
		<enhancer region index> \t <promoter region index> \t <interaction count> \n
input/<choromosome #>/<cell type>/test<dataset fold>.txt
		<enhancer region index> \t <promoter region index> \t <interaction count> \t <pairwise distance> \t <RIPPLE predicted count> \n
output/<chromosome #>_<cell type>_f<dataset fold>.hdf5
		learned cnn model saved in hdf5 format, used by load.py
output/<chromosome #>_<cell type>_f<dataset fold>.txt
		log file of cnn learning curve
output/<chromosome #>_<cell type>_f<dataset fold>_distribution.png
		histogram of interacton counts by pairwise distance
output/<chromosome #>_<cell type>_f<dataset fold>_pearson.png
		Pearson correlation coefficientby pairwise distance
output/<chromosome #>_<cell type>_f<dataset fold>_scatter.png
		scatterplot of predicted vs actual interaction counts