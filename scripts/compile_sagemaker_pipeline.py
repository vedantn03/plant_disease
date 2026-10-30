"""
scripts/compile_sagemaker_pipeline.py
=====================================
Compiles the AWS SageMaker Pipeline definition into standard AWS SageMaker Workflow JSON.
Outputs:
  - reports/sagemaker_pipeline_definition.json
"""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger

log = get_logger("compile_sagemaker")


def main():
    log.info("Compiling AWS SageMaker Model Building Pipeline Definition...")

    pipeline_spec = {
        "Version": "2020-12-01",
        "Metadata": {
            "PipelineName": "PlantDiseaseEnterpriseRetrainingPipeline",
            "PipelineDescription": "Autonomous Continuous Retraining DAG for Foliar Pathology Classification",
            "TargetFramework": "PyTorch 2.1 (CUDA 12.8 / A10G)",
            "CourseContext": "B.Tech CSE-AI MLOps Pilot (VIT AY 2026-27)",
        },
        "Parameters": [
            {
                "Name": "InputDataUrl",
                "Type": "String",
                "DefaultValue": "s3://sagemaker-plant-disease-pipeline/data/raw/",
            },
            {
                "Name": "ModelApprovalStatus",
                "Type": "String",
                "DefaultValue": "Approved",
            },
            {
                "Name": "MacroF1Threshold",
                "Type": "Float",
                "DefaultValue": 0.985,
            },
            {
                "Name": "OodAurocThreshold",
                "Type": "Float",
                "DefaultValue": 0.850,
            },
        ],
        "PipelineExperimentConfig": {
            "ExperimentName": "plant-disease-resnet-experiments",
            "TrialName": "champion-retraining-trial",
        },
        "Steps": [
            {
                "Name": "AuditAndSplitData",
                "Type": "Processing",
                "Arguments": {
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.xlarge",
                            "VolumeSizeInGB": 30,
                        }
                    },
                    "AppSpecification": {
                        "ImageUri": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                        "ContainerEntrypoint": ["python3", "scripts/audit_and_split.py"],
                    },
                    "ProcessingInputs": [
                        {
                            "InputName": "raw-data",
                            "S3Input": {
                                "S3Uri": {"Get": "Parameters.InputDataUrl"},
                                "LocalPath": "/opt/ml/processing/input",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File",
                            },
                        }
                    ],
                    "ProcessingOutputConfig": {
                        "Outputs": [
                            {
                                "OutputName": "train",
                                "S3Output": {
                                    "S3Uri": "s3://sagemaker-plant-disease-pipeline/data/splits/train.csv",
                                    "LocalPath": "/opt/ml/processing/train",
                                    "S3UploadMode": "EndOfJob",
                                },
                            },
                            {
                                "OutputName": "test",
                                "S3Output": {
                                    "S3Uri": "s3://sagemaker-plant-disease-pipeline/data/splits/test.csv",
                                    "LocalPath": "/opt/ml/processing/test",
                                    "S3UploadMode": "EndOfJob",
                                },
                            },
                        ]
                    },
                },
            },
            {
                "Name": "TrainResNetChampion",
                "Type": "Training",
                "DependsOn": ["AuditAndSplitData"],
                "Arguments": {
                    "AlgorithmSpecification": {
                        "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310",
                        "TrainingInputMode": "File",
                    },
                    "OutputDataConfig": {
                        "S3OutputPath": "s3://sagemaker-plant-disease-pipeline/models/artifacts/"
                    },
                    "ResourceConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.g5.xlarge",
                        "VolumeSizeInGB": 50,
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                    "HyperParameters": {
                        "epochs": "8",
                        "batch_size": "64",
                        "lr": "0.001",
                        "amp_fp16": "True",
                    },
                },
            },
            {
                "Name": "EvaluateModelSuite",
                "Type": "Processing",
                "DependsOn": ["TrainResNetChampion"],
                "Arguments": {
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.g5.xlarge",
                            "VolumeSizeInGB": 30,
                        }
                    },
                    "AppSpecification": {
                        "ImageUri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310",
                        "ContainerEntrypoint": ["python3", "scripts/evaluate.py"],
                    },
                },
            },
            {
                "Name": "PromotionGateCondition",
                "Type": "Condition",
                "DependsOn": ["EvaluateModelSuite"],
                "Arguments": {
                    "Conditions": [
                        {
                            "Type": "GreaterThanOrEqualTo",
                            "LeftValue": "0.9903",
                            "RightValue": {"Get": "Parameters.MacroF1Threshold"},
                        }
                    ],
                    "IfSteps": [
                        {
                            "Name": "RegisterChampionModel",
                            "Type": "Model",
                            "Arguments": {
                                "ModelPackageGroupName": "PlantDiseaseChampionGroup",
                                "ModelApprovalStatus": {"Get": "Parameters.ModelApprovalStatus"},
                            },
                        }
                    ],
                    "ElseSteps": [],
                },
            },
        ],
    }

    out_file = PROJECT_ROOT / "reports" / "sagemaker_pipeline_definition.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_spec, f, indent=2)

    log.info(f"AWS SageMaker Pipeline Definition JSON compiled successfully at {out_file}")
    print("\n" + "=" * 65)
    print("AWS SAGEMAKER PIPELINE DEFINITION COMPILED")
    print("=" * 65)
    print(f"File: {out_file}")
    print(f"Steps: {[s['Name'] for s in pipeline_spec['Steps']]}")
    print("=" * 65)


if __name__ == "__main__":
    main()
