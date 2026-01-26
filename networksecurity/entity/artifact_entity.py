from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path : str
    test_file_path : str

# # class DataIngestionConfig:
#     dataset_file_path: str = "Network_Data/phisingData.csv"
#     train_test_split_ratio: float = 0.2
#     training_file_path: str = "artifact/data_ingestion/train.csv"
#     testing_file_path: str = "artifact/data_ingestion/test.csv"

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str
    

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass 
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact