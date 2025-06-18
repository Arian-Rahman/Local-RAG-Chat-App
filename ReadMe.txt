# 🚀 Project Setup Guide :

This is an LLM powered Chatbot that leverages Locally run LLMs . 

Here's a quick guide to set up the project 

---

## ➡️ Step 1: Install Miniconda

Miniconda is a lightweight version of Anaconda that includes Conda (used to manage environments).

Download Miniconda for your system from the official site: 

https://www.anaconda.com/download

Choose based on your OS:

- Windows: `Miniconda3 Windows 64-bit`
- macOS: `Miniconda3 macOS Apple M series` or Intel version
- Linux: `.sh` installer

Install the package and verify if it's installed 

run : 
    conda --version

If installed correctly it should return you the version of conda 

## ➡️ Step 2: Setup environment


After Conda is installed , Create a new environment with python 

run: 
    conda create -n LocalLLMChatbot python
    conda activate LocalLLMChatbot

After the env has been created and activated , install pip 

run:
    conda install pip

After that navigate to the project root folder and run the following command 

run : 
    pip install -r requirements.txt


⚠️ Note: 

I have built this project on MacOs Sequoia 15.5 and installed dependencied accordingly. 
If you are runnig it on different platform and face any issue or conflict between the package versions, please contact me 


## ➡️ Step 3 : Clone Embedding Models From Hugging-Face 

Navigate to the Folder models inside the root folder of this project and clones these two embedding models directly 

⚠️ Note: Make sure to have git and git-lfs installed in your system 

run : 
    git lfs install
    git clone https://huggingface.co/BAAI/llm-embedder
    git clone https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe


## ➡️ Step 4 : Install Ollama  

Downlaod Ollama from Official website https://ollama.com/download

After you've installed it 

run : 

    ollama run deepseek-r1:14b  

exit typing  /bye in the terminal and then 

run : 

    ollama run llama3-chatqa:8b  



## ➡️ Step 5 : Run the Streamlit UI 

From the project directory open terminal and  

run : 
    streamlit run app.py

## Additional Steps 

To stop the running instance of Streamlit, in the terminal press

    control + C


To Deactivate environment

run :

    conda deactivate


