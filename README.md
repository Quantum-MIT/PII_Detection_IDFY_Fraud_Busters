# PII_Detection_IDFY_Fraud_Busters
 Personally Identifiable Information (PII) Detection
# PII Identification

 Problem Statement: 

 Develop a robust and efficient data discovery tool capable of identifying
 and classifying Personally Identifiable Information (PII) within diverse data
 repositories, including relational databases, cloud storage services (e.g.,
 Google Cloud Storage, Amazon S3) and file systems. The tool should
 accurately determine the presence and type of PII in each data point and
 subsequently assess the associated risk level for the entire database or
 object.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements

```bash
pip install -r requirements.txt
```
## Features
* PII Detection: Analyze text or files for PII like names, emails, and phone numbers.
* Risk Scoring: Provides a risk score based on the detected PII.
* AWS Integration: Optional AWS S3 credentials for storing and accessing data.
* MySQL Integration: Connects to a MySQL database to store or retrieve PII data.

## Usage
Has a total of 3 backends and 2 Flask frontends. 

backend.py => Supports BERT Model and Regex Pattern Matching for PII Identification. It calls index.html which supports local file upload and AWS S3 file upload Support. In line 116,117 of bucket name and file name can be mentioned
```bash
python backend.py
```
backend2.py => Supports Classification Models like Logistic Regression with TF/IDF Vectorizer and Regex Pattern Matching for PII Identification. It calls index.html which supports local file upload and AWS S3 file upload Support.
```bash
python backend2.py
```

backend3.py => Same as backend.py with an additional support of MySQL. It calls index2.html
```bash
python backend3.py
```
## Demo [Important]
* For a clean demo, use the AWS Access keys [non-vulnerable keys] provided in the config.json file
* Go to http://127.0.0.1:5000 to use the app [or whatever shows localhost shows in the terminal]
* backend.py and backend2.py local file and AWS S3 support:
![PII Risk Scoring](https://drive.google.com/uc?export=view&id=1nrgyHr0EZNMhhKjDPYJVeYHD3OEcMxpt)
* backend3.py AWS S3 and MYSQL support:
![PII Risk Scoring](https://drive.google.com/uc?export=view&id=1pQ_hvsyaGfzPUWBiubZXwyGzJWGhmvI4)


## License

[MIT](https://choosealicense.com/licenses/mit/)
