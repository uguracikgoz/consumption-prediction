# Pod Consumption Prediction Service

## Overview
This service predicts frontend and backend pod requirements based on business metrics using machine learning. It leverages Google Sheets as a data source and provides predictions via a REST API.

## Prerequisites

- Python 3.12
- Docker
- Google Cloud Account
- Google Sheets API
- Create .src/.env file out of .src/.env.template
- Create .src/google-creds.json file out of google-test-account-service.json

## Quick Start

### Docker Deployment
The easiest way to get started is using Docker:

```bash
# Clone the repository
git clone https://github.com/uguracikgoz/consumption-prediction.git

# Navigate to the prediction service
cd consumption-prediction/prediction-service

# Build and run the Docker container
sh ./src/build_docker.sh
```

Once running, access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

### Kubernetes Deployment
Kubernetes manifests are available in the `./k8s` directory for production deployments.

## Authentication

**Note**: The included credentials are for demonstration purposes only. They:
- Only provide access to dummy data in Google Sheets
- Should be replaced with your own credentials for production use

## Model Training

### Data Requirements
The prediction model is trained on historical business metrics and pod consumption data. For accurate predictions:

- Use real production data for training when possible
- The more data points available, the better the prediction accuracy
- Ensure data includes various business cycles and seasonality patterns
- Small datasets (e.g., <50 rows) require more conservative hyperparameters to avoid overfitting

### Feature Engineering
Feature engineering is the critical foundation of effective ML modeling, particularly for this pod prediction system. The model adapts to different business scenarios through domain-specific feature creation:

#### Base Features
- Business metrics normalized by user count (`gmv_per_user`, `marketing_per_user`)
- Time-based features (day of week, month, week of year, weekend flag)

#### Advanced Transformations
- Non-linear user transformations (square root, logarithmic, squared) to capture scaling effects
- Normalized versions of key metrics to balance feature importance

#### Interaction Features
- User-GMV product captures combined traffic effects
- Weekend-specific user patterns to identify demand fluctuations
- Day-of-week user interactions to model weekly usage patterns

#### Traffic Modeling
- Traffic index combining user count with per-user value
- Peak factor incorporating marketing-driven traffic spikes

The feature engineering approach can be fine-tuned based on specific business needs:
- E-commerce sites might prioritize GMV-related features
- Content platforms might emphasize user engagement metrics
- B2B services might focus on workday patterns and resource intensity

### Algorithm
The service uses Gradient Boosting Regression with hyperparameters optimized for small datasets:

- Conservative parameters to prevent overfitting:
  - `n_estimators`: 100 (reduced tree count)
  - `max_depth`: 3 (shallow trees)
  - `learning_rate`: 0.05 (slow learning rate)
  - `subsample`: 0.8 (training on 80% sample for each tree)
- Features ranked by importance for model transparency
- Non-shuffled train/test split to preserve time series patterns

### Evaluation
Models are evaluated using multiple metrics:

- Mean Absolute Error (MAE): Intuitive measure of prediction error
- Root Mean Squared Error (RMSE): Penalizes larger errors
- Separate evaluation on training and test sets to detect overfitting
- Correlation analysis between features and targets to identify key relationships

## Caching
Predictions are cached for 7 days by default to improve response times and reduce computational load. You can modify the cache TTL in the `.env` file.

## Project Structure

```
prediction-service/
│
├── src/                     # Source code
│   ├── api/                 # API endpoints
│   ├── data/                # Data loaders
│   ├── model/               # ML models
│   ├── cron/                # Scheduled jobs
│   ├── init_model.py        # Model initialization
│   └── build_docker.sh      # Docker build script
│   └── run_docker.sh        # Docker run script
│   └── run_api.py           # API run script
│   └── requirements.txt     # Python dependencies
│   └── Dockerfile           # Dockerfile
│   └── .env                 # Environment variables
│
├── k8s/                     # Kubernetes manifests
└── .env                     # Environment configuration
```