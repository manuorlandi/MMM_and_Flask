# Flask web app che espone api REST

I file vengono creati con il notebook EDA.ipynb.
All'interno del file di configurazione conf.yml è necessario andare a cambiare il nome del file da analizzare (di facebook, google e amazon), settando il parametro FILE_TO_EXPLORE.
Il notebook produce come output un csv pulito, l'output prende il nome dal parametro OUTPUT_FILE.
EDA_fin.ipynb è un notebook che ho utilizzato per andare ad esplorare il dataset finale, joinato con le revenues.

Una volta creati i 3 file necessari si può passare alla fase di docker.

# Docker phase

Assicurarsi che il docker deamon stia girando ed esegui dentro la folder principale

> `docker build -t ml-model .`

Subito dopo far partire un container con 
    
> `docker run -d -p 5000:5000 ml-model`

controllare che il container stia girando con 

> `docker ps`

La web app espone 2 servizi, train e results, inerrogabili tramite 'http://localhost:5000/train' e 'http://localhost:5000/results'
oppure ad esempio tramite curl

![Screenshot](img/image.png)

> `curl -X GET http://localhost:5000/train`   