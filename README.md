# pedyfun-NLP
NLP project on SemEval-2020 task-7

The main idea of the my protocol is to find the word embeddings of the sentences using BERT or its derivatives and use CNNs to map them to a grade range of 0-3:
	It uses BERT, AlBERT, RoBERTa, and DistilBERT separately to find the word-embeddings of each pair of 	edied/original sentence.

There are 4 pthon codes:

Task1.py:	has the code for task 1. At the end of the run it shows the results for using each one of the BERT or its derivatives methods. It also saves the best CNN models found and their result on the test-set.

Task2.py:	has the code for task 2. At the end of the run it shows the results for using each one of the BERT or its derivatives methods. It also saves the best CNN models found and their result on the test-set. Here, the dataset includes two edits for each original sentence. The word-embeddings of each pair of original/edit 1 and original/edit 2 is given to a CNN layer, and the results are concatenated and given to a dense layer to decide which one is more funny.

BothTasks:	has the code for task 1 followed by a code for task 2 that only uses the best models found in task 1 to grade each one of the edits in the pairs separatly, and then, compare them together.

TestResults:	has the code for comparing the results of each task with the expected grades/labels of the test-sets in each task. This code uses the two functions given by the dataset to find the RMSE in task 1 and Accuracy in task 2.

	After either the first 2 codes or the 3rd code are run completely, the last code (TestResults.py) can be run to test their results. Or the already available final results can go through this code to see their results.


Run me instructions:
	
	1. put all the codes and data folder in a directory and use terminal to go to the directory
	
	2. make sure python(>3.5) is installed 
	
	3.	a. make a Virtualenv by: apt-get install python-virtualenv
		b. or install the Virtualenv available here by: source env create -f pedy.yml
	
	4. activate the virtual env by: source bin/activate
	
	5. use pip to install the requirements in INSTALL.txt by: pip install INSTALL.txt
	
	6. in case of having an issue installing torch, use the directions at https://pytorch.org/get-started/locally/
	
	7. in case of having an issue installing transformers together with torch use: pip install transformers
	
	8. make sure to install transformers 3.1.0 or newer to include all BERT, AlBERT, RoBERTa, and DistilBERT
	
	9. run either one of the codes Task1.py, Task2.py, or BothTasks.py
	
	10. run TestResults.py to see the final results of previous codes
