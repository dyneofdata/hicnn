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
output/\[chromosome #\]\_\[cell type\]\_f\[dataset fold\].hdf5 | learned cnn model saved in hdf5 format, used by load.py
output/\[chromosome #\]\_\[cell type\]\_f\[dataset fold\].txt | log file of cnn learning curve
output/\[chromosome #\]\_\[cell type\]\_f\[dataset fold\]\_distribution.png | histogram of interacton counts by pairwise distance
output/\[chromosome #\]\_\[cell type\]\_f\[dataset fold\]\_pearson.png  | Pearson correlation coefficient by pairwise distance
output/\[chromosome #\]\_\[cell type\]\_f\[dataset fold\]\_scatter.png  | scatterplot of predicted vs actual interaction counts
