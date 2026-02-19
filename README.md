# Smart Health Assist

This git repo presents Smart Health Assist, a web-based intelligent medical chatbot designed to assist patients in managing their healthcare needs and accessing trustworthy medical information through conversational AI and retrieval-augmented generation (RAG). 

The system integrates the Mistral large language model (LLM) hosted on Ollama with ChromaDB to retrieve relevant medical knowledge and generate context-grounded responses for queries related to symptoms, health guidance, and medication. Alongside conversational support, it provides secure storage and retrieval of electronic medical records (EMRs) such as prescriptions and diagnostic reports, and enables appointment scheduling with nearby hospitals. 

The system adopts a FastAPI-based microservice architecture distributed across two laptops: 
1. one hosting the React-based frontend for user interaction and EMR management, 
2. and the other running the Ollama inference engine and ChromaDB retrieval backend for computational processing. 

This separation improves efficiency, scalability, and resource utilization. 

Prototype testing demonstrates an average response latency of 7.8 seconds per query. The results validate the feasibility of integrating retrieval-augmented conversational AI with secure health data management to deliver a practical, patient-centric assistant that enhances healthcare accessibility, reduces administrative workload, and supports informed decision-making in clinical contexts.