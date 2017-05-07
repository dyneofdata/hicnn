## Convolutional Neural Network (CNN) for HI-C Interaction Count Prediction

### Usage 
python hicnn.py \[chromosome #\] \[cell type\]
```
python cnn.py 1 Gm12878
```
### Input Files
- input/\[chromosome #\]/raw_sequence.txt
  > chr1_5000_10000	CACTGTAAAAATGGAAGTAATTCCCATTAGGACCAGCAAAACCTGAGGCTAAAAAAAGACAGTAAAAGCTCATGCCAAAAGCTGAATTTTACTTAATATAAAGAAAGGTGGCAGTTTCCAATTTCAGTAGAAAGTAGGAGTGTCAAATTGCTACAGAAACTGCCATCCTCCAGAGACTGACGACCCGAATGAACCCAGAGGCAATTTTTTATTCTCATGAGATGGCTTGCTTAGATATTTCTGGGAAGGAGCAGTAGGTCTTAGGAAAGGTTAGAATGTTGTTGTTTCCTGGTAACTACTTGCAGAGGTTGATAGGAGTCAATGAGACCAA...
- input/\[choromosome #\]/\[cell type\]/markers.txt (Currently not in use as of 5/7/2017)
  > \[region index\] \t \[chromatin marker signal 1\] \t \[chromatin marker signal 2\] \t ... \n
- input/\[choromosome #\]/\[cell type\]/train\[dataset fold\].txt
  > \[enhancer region index\] \t \[promoter region index\] \t \[interaction count\] \n
- input/\[choromosome #\]/\[cell type\]/test\[dataset fold\].txt
  > \[enhancer region index\] \t \[promoter region index\] \t \[interaction count\] \t \[pairwise distance\] \t \[RIPPLE predicted count\] \n

### Output Files
File | Description
-----|------------
output/\[chromosome #\]\_\[cell type\].hdf5 | learned model saved in hdf5 format
output/\[chromosome #\]\_\[cell type\].txt | log file of learning curve
output/\[chromosome #\]\_\[cell type\]\_distribution.png | mean and standard deviation of actual counts plotted next to those of predicted counts, by pairwise distance
output/\[chromosome #\]\_\[cell type\]\_pearson.png  | Pearson correlation coefficient between predicted and actual counts, by pairwise distance
output/\[chromosome #\]\_\[cell type\]\_scatter.png  | scatterplot of predicted vs actual interaction counts
