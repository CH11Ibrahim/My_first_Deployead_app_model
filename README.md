# My_first_Deployed_Model
Streamlit site to show the results of traning model

# Trane AI Models for Intrusion Detection in Wireless Sensor Networks
## Overview
This repository contains AI models designed for the detection of various types of attacks in Wireless Sensor Networks (WSN), including Blackhole, Grayhole, Flooding,TDMA (Time Division Multiple Access) attack, and normal network states.

## Resume
### About the Project
- **Purpose**: To identify and classify attack types in WSN using the WSN-DS dataset. 
- **Technologies Used**: Python, Scikit-learn, Jupyter Notebook.

### Key Features
- Detects Blackhole, Grayhole, Flooding attacks, and normal states.
- Utilizes the WSN-DS dataset from Kaggle for training and validation.
- Implements machine learning algorithms for classification.

### Authors
- **Chamanamas Ibrahim** -Developer

## Dataset
The models are trained using the **WSN-DS** dataset, which includes the following attributes:

- **Node ID**: Unique ID for each sensor node.
- **Time**: Current simulation time.
- **Is CH?**: Flag indicating if the node is a Cluster Head (1) or normal (0).
- **Who CH?**: ID of the Cluster Head in the current round.
- **RSSI**: Received Signal Strength Indication.
- **Distance to CH**: Distance between the node and its Cluster Head.
- **Max distance to CH**: Maximum distance within the cluster.
- **Average distance to CH**: Average distance of nodes to their Cluster Head.
- **Current energy**: Current energy level of the node.
- **Energy consumption**: Energy consumed in the previous round.
- **ADV_CH send/receive**: Number of advertisement messages sent/received.
- **Join_REQ send/receive**: Number of join request messages sent/received.
- **ADV_SCH send/receive**: Number of TDMA schedule messages sent/received.
- **Rank**: Order of the node within the TDMA schedule.
- **Data sent/received**: Number of data packets sent/received.
- **Data sent to BS**: Number of packets sent to the Base Station.
- **Distance CH to BS**: Distance between the Cluster Head and the Base Station.
- **Send Code**: Cluster sending code.
- **Attack Type**: Class of attack (Blackhole, Grayhole, Flooding,TDMA attack, Normal).
## Note :
multi-attack-models containes models thats trained to detect the attack type and the file named ids-detection-models gives you detection if there is a threat or not on your WSN
## Getting Started

### Prerequisites
- Docker installed on your machine.
- Basic understanding of Docker commands.

### Steps to Run the Application

1. **Clone the Repository**
   ```bash
   git clone https://github.com/CH11Ibrahim/My_first_Deployead_app_model.git
   cd My_first_Deployead_app_model
2. **Build a docker image**
   ```bash
   docker build -t trane-ai-models .
   
3. **Run the Docker Container**
   ```bash
   docker run -p 8888:8888 trane-ai-models
# Access the Streamlit App
Open your web browser and go to https://myfirstdeployeadappmodel.streamlit.app/ to view the application. 
# Acknowledgments
WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks.
# Take a look of the work online :
https://myfirstdeployeadappmodel.streamlit.app/
