# Star_Detection

## SARA

**Task 1** *consegna 12/12*

Completa Star Detection per casistiche
* Dark current
* Blur
* Blur current noise

**Task 2** *consegna 20/12*

Scrivi algoritmo completo e calcola coordinate
*   Parti da studio luminosità
*   Data ogni immagine viene ritornato un vettore di coordinate (x,y) per la loro localizzazione

## Andreea

**Task 1** *consegna 12/12*

Crea dataset
* Riferimento a parametri elencati pipeline-options.hpp nella cartella lost/src
  
  Per avere una base: le immagini di prova sono state generate variando i parametri del comando:
     ```bash
  ./lost pipeline   --generate 1   --generate-x-resolution 1024   --generate-y-resolution 1024   --fov 30  --generate-spread-stddev 1
  --generate-read-noise-stddev 0.05   --generate-ra 88   --generate-de 7   --generate-roll 0  --    generate-dark-current 0.0
  --generate-blur-de 1.5 --generate-blur-ra 1.8 --generate-blur-roll 1.9  --plot-raw-input raw-input.png   --plot-input annotated-input.png 
     ```
  Ogni immagine creata con il rumore ha bisogno di essere accompagnata dalla quantità di stelle contenute, per fare questo potrebbe essere utile:
  * generare due immagini con stessa posizione e orientamento di cui una pulita e una con rumore
  * passare l'immagine pulita dal rumore nell'algoritmo e associare il numero di stelle rilevate a quella rumorosa
  * eliminare le immagini pulite e mantenere esclusivamente quelle ruomorose e il numero di stelle in esse contenute
  Nota: la parte di algoritmo necessaria a fare questo è già stata sviluppata e si trova nella sezione ""Analyzing the perfect image"

**Task 2** *consegna da definire*

Calcola centroide stelle
* Riferimento al paper [High-Precision Centroid Localization Algorithm](https://www.mdpi.com/2072-4292/17/7/1108#)
* Per un calcolo più accurato, potrebbe tornare utile prendere in causa un regione circostante le coordinate riportate come risultato dell'algoritmo di detezione
