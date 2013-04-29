Setup and Requirements
===========================

**Requirements:**
  
  * Python >= 2.7 
  
  This project depends on some 3rd party apps.
  
  * matplotlib >= 1.1.0  
  * nltk >= 2.0.4  
  * numpy >= 1.6.1  
  * gensim >= 0.8.6

**Setup**

  To run the program, we need to get the answer log from WebWorks.
  
  The name of this file is usually answer_log and it contains the answers
  from the students for the quizzes that were set up on webworks. 
  
  Put this file in the directory where you put the JITT python code.
  

**Running the Program**

  When you have the answer log in the directory, run the following: ::
    
    $ python ./lsi_JIT_main.py answer_log Lecture_4 3
   
  * **./lsi_JIT_main.py** - Run lsi_JIT_main.py that is currently in this directory.
  * **answer_log** - The name of the answer log.
  * **Lecture_4**  - The name of the quiz that we are interested in harvesting.
  * **3**          - Is the rank of the SVD matrix we would like to reconstruct.