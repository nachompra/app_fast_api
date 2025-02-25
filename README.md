# app_fast_api
### PASOS
1. docker pull nachompra/mi_api:v1
2. docker run -p 8000:8000 nachompra/mi_api:v1
### TEST
1. predict
localhost/8000/predict  
Usa este json  
{  
  "tv": 100.0,  
  "radio": 20.0,  
  "newspaper": 10.0   
}  
