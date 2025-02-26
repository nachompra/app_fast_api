# app_fast_api
### PASOS
1. docker pull nachompra/mi_api:v2
2. docker run -p 8000:8000 nachompra/mi_api:v2
### PRUEBAS
1. http://127.0.0.1:8000/v1/predict
   POST->BODY: {"data": [[100, 100, 200]]}
2. http://127.0.0.1:8000/ingest
   POST->BODY: {"data": [[100, 100, 200, 3000], [200, 230, 500, 4000]]}
