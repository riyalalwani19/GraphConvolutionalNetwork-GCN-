
Files
•	Main.py – Web app file		
•	gnn_nodeclassification.py  – Model file		
•	requirements.txt  – Package Installation requirements		
•	Dockerfile  – file required for docker 		
•	Cora folder – dataset 		
•	docker-compose.yml  – Docker compose config file		
•	Model.pkl – Trained Model pickle file		


Steps to recreate: 
1.	Getting the code.
a.	Download the code from code.zip file attached, or code is also uploaded on Github (link) along with dataset
		
2.	Installing the required packages from requirement.py file and few packages are only available on URL as mentioned below:
a.	pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
b.	pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
c.	cu102 will change according to cuda version installed on your machine.

3.	Running the code on local server.
a.	Create project with cora dataset, model.pkl, and main.py.
b.	Change path locations for dataset and pickle file according to your system.
c.	Run  main.py file.
d.	Open the URL (http://localhost:8080/docs) and give input between 0-2708 to get the desired output stating category of the paper whose ID was provided. 

4.	Creating Docker image on local machine and pushing the image on docker hub.
a.	docker login with your credentials.
b.	Create repository
c.	docker build -t dockerfor2021(docker hub name)/ graphneuralnetworks (repository name):latest .
d.	docker push dockerfor2021(docker hub name)//graphneuralnetworks (repository name):latest

5.	Azure cloud deployment
a.	Create repository and web app service.
b.	For web app create resource group, Instance name and for publishing select Docker container, Operating system – Linux, Region –  Northern Europe.

 
 

c.	Next Docker – Select Options as Docker Compose, Image source  as Docker Hub and configuration file as docker-compose.yml.
d.	Docker-compose contains docker image mapping and running image code.
e.	Review and create service.
f.	Start the service and API link is generated as below and can be accessed from anywhere.: 
https://graph-predict.azurewebsites.net/docs


