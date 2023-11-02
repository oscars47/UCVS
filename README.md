# UCVS
![tde (1)](https://github.com/oscars47/UCVS/assets/106777951/7721f42e-6275-4127-8901-b0066e5a2fad)

Welcome all ! This is a repo for the Fall 2023 P-ai project (PMs: Oscar Scholin and Graham Hirsch) to build an unsupervised clustering model to classify variable stars. The code is organized as follows: 

`./processing' has the code to process the lightcurves; 'mmgen2.py' is a master file that applies the variability indices calculations stored in the Variable2 class in 'vargen2.py' as well as  color calculations; all of this information is stored in a main csv called 'mm_(u)n.csv', aka the "Monster Matrix". Also in './processing' is 'lcgen.py', which allows one to quickly plot the folded and unfolded lightcurves given an object ID in the ASAS-SN catalog. Note the processed data generated during Fall 2022 can be found [here]([url](https://drive.google.com/drive/folders/1PgVBjVWzdmSGbx42nixHeabedoXOK9Cl)).

'./supervised' has the code from Fall 2022 in which we convert the csv NAME into a numpy array along with a one-hot encoded array of the labels in 'nn_prep.py', which we feed into a neural network 'nn.py' to classify the variable stars. We can then test the neural network in 'nn_predict.py'. For a tutorial on neural networks, see the subdirectory 'tutorial' and the Jupyter notebook 'Iris_forward.ipynb'.

'./unsupervised' has the old code from Fall 2022 in the subdirectory './old' and a folder 'p-ai' for the code for the unsupervised models for this project. The sub-subdirectories './indices' and './tree' are for those methods, respectively.
