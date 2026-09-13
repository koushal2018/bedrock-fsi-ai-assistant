Scenario

You're building a customer service AI assistant for a financial services company. The assistant needs to:

Answer product questions based on company documentation

Generate personalized responses to customer inquiries

Maintain high availability and consistent performance

Comply with financial industry regulations

Part 1: Foundation Model assessment and benchmarking

Set up a benchmarking framework to evaluate different Amazon Bedrock models

Compare models based on:

Response quality for financial domain questions

Latency and throughput

Cost per request

Compliance with guardrails

Step 2. Flexible architecture for dynamic model selection

Implement a model abstraction layer using AWS Lambda

Configure AWS AppConfig for dynamic model selection rules

Create an API Gateway endpoint for consistent client access

Step 3. Resilient system design

Implement AWS Step Functions with circuit breaker patterns

Set up cross-region model deployment for high availability

Create graceful degradation strategies for service disruptions

Step 4. Model customization and lifecycle management

Fine-tune a model for financial domain using Amazon SageMaker

Implement model versioning and deployment workflows

Create automated testing and rollback strategies

Implementation examples

Part 1: Foundation Model assessment and benchmarking

Step 1: Set up evaluation framework

Create a Python script to evaluate models using Amazon Bedrock:


import boto3
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Initialize Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime')

# Models to evaluate
models = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-instant-v1",
    "amazon.titan-text-express-v1"
]

# Test cases with ground truth answers
test_cases = [
    {
        "question": "What is a 401(k) retirement plan?",
        "context": "Financial services",
        "ground_truth": "A 401(k) is a tax-advantaged retirement savings plan offered by employers."
    },
    # Add more test cases...
]

def invoke_model(model_id, prompt, max_tokens=500):
    """Invoke a model with the given prompt and return the response and metrics."""
    start_time = time.time()
    
    # Prepare request body based on model provider
    if "anthropic" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    elif "amazon" in model_id:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": 0.7,
                "topP": 0.9
            }
        })
    # Add more model providers as needed
    
    try:
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read().decode())
        
        if "anthropic" in model_id:
            output = response_body['content'][0]['text']
        elif "amazon" in model_id:
            output = response_body['results'][0]['outputText']
        
        # Calculate metrics
        latency = time.time() - start_time
        token_count = len(output.split())  # Rough estimate
        
        return {
            "success": True,
            "output": output,
            "latency": latency,
            "token_count": token_count
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.time() - start_time
        }

def evaluate_models():
    """Evaluate all models on all test cases and return results."""
    results = []
    
    for test_case in test_cases:
        prompt = f"Question: {test_case['question']}\nContext: {test_case['context']}"
        
        for model_id in models:
            print(f"Evaluating {model_id} on: {test_case['question']}")
            response = invoke_model(model_id, prompt)
            
            if response["success"]:
                # Calculate similarity score with ground truth (simplified)
                similarity = calculate_similarity(response["output"], test_case["ground_truth"])
                
                results.append({
                    "model_id": model_id,
                    "question": test_case["question"],
                    "output": response["output"],
                    "latency": response["latency"],
                    "token_count": response["token_count"],
                    "similarity_score": similarity
                })
            else:
                results.append({
                    "model_id": model_id,
                    "question": test_case["question"],
                    "error": response["error"],
                    "latency": response["latency"]
                })
    
    return pd.DataFrame(results)

def calculate_similarity(output, ground_truth):
    """Calculate similarity between model output and ground truth (simplified)."""
    # In a real implementation, use more sophisticated NLP techniques
    # This is a very simplified version
    output_words = set(output.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    if not truth_words:
        return 0.0
        
    common_words = output_words.intersection(truth_words)
    return len(common_words) / len(truth_words)

# Run evaluation
if __name__ == "__main__":
    results_df = evaluate_models()
    
    # Save results to CSV
    results_df.to_csv("model_evaluation_results.csv", index=False)
    
    # Print summary
    print("\nEvaluation Summary:")
    summary = results_df.groupby("model_id").agg({
        "latency": "mean",
        "similarity_score": "mean",
        "token_count": "mean"
    }).reset_index()
    
    print(summary)
Step 2: Analyze results and create a model selection strategy

Create a selection strategy based on your benchmark results:


def create_model_selection_strategy(results_df):
    """Create a model selection strategy based on evaluation results."""
    # Calculate overall scores
    model_scores = results_df.groupby("model_id").agg({
        "latency": "mean",
        "similarity_score": "mean"
    }).reset_index()
    
    # Normalize scores (lower latency is better, higher similarity is better)
    max_latency = model_scores["latency"].max()
    model_scores["latency_score"] = 1 - (model_scores["latency"] / max_latency)
    
    # Calculate weighted score (adjust weights based on priorities)
    model_scores["overall_score"] = (
        0.7 * model_scores["similarity_score"] + 
        0.3 * model_scores["latency_score"]
    )
    
    # Sort by overall score
    model_scores = model_scores.sort_values("overall_score", ascending=False)
    
    # Create strategy
    strategy = {
        "primary_model": model_scores.iloc[0]["model_id"],
        "fallback_models": model_scores.iloc[1:]["model_id"].tolist(),
        "model_scores": model_scores.to_dict(orient="records")
    }
    
    return strategy

# Generate strategy
strategy = create_model_selection_strategy(results_df)
print(json.dumps(strategy, indent=2))

# Save strategy to file for AppConfig
with open("model_selection_strategy.json", "w") as f:
    json.dump(strategy, f, indent=2)
Part 2: Flexiable architecture for dynamic model selection

Step 1: Create AWS AppConfig for model configuration

Set up AWS AppConfig using AWS CLI or console:


# Create application
aws appconfig create-application --name "AIAssistantApp"

# Create environment
aws appconfig create-environment --application-id YOUR_APP_ID --name "Production"

# Create configuration profile
aws appconfig create-configuration-profile \
    --application-id YOUR_APP_ID \
    --name "ModelSelectionStrategy" \
    --location-uri "hosted" \
    --type "AWS.AppConfig.FeatureFlags"

# Create and deploy configuration
aws appconfig create-hosted-configuration-version \
    --application-id YOUR_APP_ID \
    --configuration-profile-id YOUR_PROFILE_ID \
    --content-type "application/json" \
    --content file://model_selection_strategy.json

# Deploy configuration
aws appconfig start-deployment \
    --application-id YOUR_APP_ID \
    --environment-id YOUR_ENV_ID \
    --configuration-profile-id YOUR_PROFILE_ID \
    --configuration-version 1 \
    --deployment-strategy-id YOUR_STRATEGY_ID
Step 2: Create model abstraction Lambda

Create a Lambda function for model abstraction:


import boto3
import json
import os

def lambda_handler(event, context):
    # Get AppConfig configuration
    appconfig_client = boto3.client('appconfig')
    config_response = appconfig_client.get_configuration(
        Application='AIAssistantApp',
        Environment='Production',
        Configuration='ModelSelectionStrategy',
        ClientId='AIAssistantLambda'
    )
    
    # Parse configuration
    config = json.loads(config_response['Content'].read().decode('utf-8'))
    
    # Extract request details
    body = json.loads(event.get('body', '{}'))
    prompt = body.get('prompt', '')
    use_case = body.get('use_case', 'general')
    
    # Select model based on use case and configuration
    model_id = select_model(config, use_case)
    
    # Invoke selected model
    response = invoke_model(model_id, prompt)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'model_used': model_id,
            'response': response
        })
    }

def select_model(config, use_case):
    """Select appropriate model based on configuration and use case."""
    # Check if there's a use case specific model
    use_case_models = config.get('use_case_models', {})
    if use_case in use_case_models:
        return use_case_models[use_case]
    
    # Default to primary model
    return config.get('primary_model')

def invoke_model(model_id, prompt):
    """Invoke the selected model with error handling."""
    bedrock_runtime = boto3.client('bedrock-runtime')
    
    try:
        # Prepare request body based on model provider
        if "anthropic" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        elif "amazon" in model_id:
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 500,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        # Add more model providers as needed
        
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read().decode())
        
        if "anthropic" in model_id:
            return response_body['content'][0]['text']
        elif "amazon" in model_id:
            return response_body['results'][0]['outputText']
        
    except Exception as e:
        print(f"Error invoking model {model_id}: {str(e)}")
        # Return error message or try fallback model
        return f"Error generating response: {str(e)}"
Step 3: Set Up API Gateway

Create an API Gateway REST API:


# Create API
aws apigateway create-rest-api --name "AIAssistantAPI"

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources --rest-api-id YOUR_API_ID --query 'items[0].id' --output text)

# Create resource
aws apigateway create-resource --rest-api-id YOUR_API_ID --parent-id $ROOT_ID --path-part "generate"

# Create POST method
aws apigateway put-method --rest-api-id YOUR_API_ID --resource-id YOUR_RESOURCE_ID --http-method POST --authorization-type "NONE"

# Set up Lambda integration
aws apigateway put-integration --rest-api-id YOUR_API_ID --resource-id YOUR_RESOURCE_ID --http-method POST --type AWS_PROXY --integration-http-method POST --uri "arn:aws:apigateway:REGION:lambda:path/2015-03-31/functions/YOUR_LAMBDA_ARN/invocations"

# Deploy API
aws apigateway create-deployment --rest-api-id YOUR_API_ID --stage-name "prod"
Part 3: Resilient system design

Step 1: Create Step Functions workflow with circuit breaker

Create a Step Functions state machine definition:


{
  "Comment": "AI Assistant with Circuit Breaker Pattern",
  "StartAt": "TryPrimaryModel",
  "States": {
    "TryPrimaryModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Parameters": {
        "FunctionName": "${PrimaryModelLambdaArn}",
        "Payload": {
          "prompt.$": "$.prompt",
          "use_case.$": "$.use_case"
        }
      },
      "Retry": [
        {
          "ErrorEquals": ["States.TaskFailed"],
          "IntervalSeconds": 1,
          "MaxAttempts": 2,
          "BackoffRate": 2
        }
      ],
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "Next": "TryFallbackModel"
        }
      ],
      "Next": "SuccessState"
    },
    "TryFallbackModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Parameters": {
        "FunctionName": "${FallbackModelLambdaArn}",
        "Payload": {
          "prompt.$": "$.prompt",
          "use_case.$": "$.use_case",
          "is_fallback": true
        }
      },
      "Retry": [
        {
          "ErrorEquals": ["States.TaskFailed"],
          "IntervalSeconds": 1,
          "MaxAttempts": 2,
          "BackoffRate": 2
        }
      ],
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "Next": "GracefulDegradation"
        }
      ],
      "Next": "SuccessState"
    },
    "GracefulDegradation": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Parameters": {
        "FunctionName": "${DegradationLambdaArn}",
        "Payload": {
          "prompt.$": "$.prompt",
          "use_case.$": "$.use_case"
        }
      },
      "Next": "SuccessState"
    },
    "SuccessState": {
      "Type": "Succeed"
    }
  }
}
Create the fallback model Lambda:


import boto3
import json

def lambda_handler(event, context):
    """Fallback model handler that uses a simpler, more reliable model."""
    prompt = event.get('prompt', '')
    use_case = event.get('use_case', 'general')
    
    # Use a simpler, more reliable model
    model_id = "amazon.titan-text-express-v1"  # Example fallback model
    
    try:
        # Invoke the model with simplified parameters
        bedrock_runtime = boto3.client('bedrock-runtime')
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 300,  # Reduced for reliability
                    "temperature": 0.5,
                    "topP": 0.9
                }
            })
        )
        
        response_body = json.loads(response['body'].read().decode())
        output = response_body['results'][0]['outputText']
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'model_used': f"FALLBACK:{model_id}",
                'response': output
            })
        }
    except Exception as e:
        # Let Step Functions catch this and move to graceful degradation
        raise Exception(f"Fallback model failed: {str(e)}")
Create the graceful degradation Lambda:


import json

def lambda_handler(event, context):
    """Graceful degradation handler that returns a predefined response."""
    prompt = event.get('prompt', '')
    use_case = event.get('use_case', 'general')
    
    # Provide a graceful response based on the use case
    responses = {
        "general": "I'm sorry, but I'm currently experiencing technical difficulties. Please try again later or contact customer service for immediate assistance.",
        "product_question": "I apologize, but I can't access product information right now. Please refer to our product documentation or contact customer service at 1-800-555-1234.",
        "account_inquiry": "I'm unable to process account inquiries at the moment. For urgent matters, please call our customer service line at 1-800-555-1234."
    }
    
    default_response = "I'm sorry, but I'm currently experiencing technical difficulties. Please try again later."
    response_text = responses.get(use_case, default_response)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'model_used': "DEGRADED_SERVICE",
            'response': response_text
        })
    }
Step 2: Set up cross-Region deployment

Create a CloudFormation template for cross-Region deployment:


AWSTemplateFormatVersion: '2010-09-09'
Description: 'AI Assistant Cross-Region Deployment'

Parameters:
  Environment:
    Type: String
    Default: prod
    AllowedValues:
      - dev
      - prod
    Description: Deployment environment

Resources:
  ModelAbstractionLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub "ai-assistant-model-abstraction-${Environment}"
      Handler: index.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Runtime: python3.9
      Timeout: 30
      MemorySize: 256
      Code:
        ZipFile: |
          import boto3
          import json
          import os
          
          def lambda_handler(event, context):
              # Implementation as shown earlier
              pass
  
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: 'sts:AssumeRole'
      ManagedPolicyArns:
        - 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
      Policies:
        - PolicyName: BedrockAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - 'bedrock:InvokeModel'
                  - 'bedrock-runtime:InvokeModel'
                Resource: '*'
        - PolicyName: AppConfigAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - 'appconfig:GetConfiguration'
                Resource: '*'
  
  ApiGateway:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: !Sub "ai-assistant-api-${Environment}"
      Description: API for AI Assistant
  
  ApiResource:
    Type: AWS::ApiGateway::Resource
    Properties:
      RestApiId: !Ref ApiGateway
      ParentId: !GetAtt ApiGateway.RootResourceId
      PathPart: "generate"
  
  ApiMethod:
    Type: AWS::ApiGateway::Method
    Properties:
      RestApiId: !Ref ApiGateway
      ResourceId: !Ref ApiResource
      HttpMethod: POST
      AuthorizationType: NONE
      Integration:
        Type: AWS_PROXY
        IntegrationHttpMethod: POST
        Uri: !Sub "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${ModelAbstractionLambda.Arn}/invocations"
  
  ApiDeployment:
    Type: AWS::ApiGateway::Deployment
    DependsOn: ApiMethod
    Properties:
      RestApiId: !Ref ApiGateway
      StageName: !Ref Environment

Outputs:
  ApiEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ApiGateway}.execute-api.${AWS::Region}.amazonaws.com/${Environment}/generate"
  LambdaArn:
    Description: Lambda function ARN
    Value: !GetAtt ModelAbstractionLambda.Arn
Deploy the CloudFormation template to multiple Regions:


# Deploy to primary region
aws cloudformation deploy \
    --template-file template.yaml \
    --stack-name ai-assistant-stack \
    --parameter-overrides Environment=prod \
    --region us-east-1 \
    --capabilities CAPABILITY_IAM

# Deploy to secondary region
aws cloudformation deploy \
    --template-file template.yaml \
    --stack-name ai-assistant-stack \
    --parameter-overrides Environment=prod \
    --region us-west-2 \
    --capabilities CAPABILITY_IAM
Set up Route 53 for cross-Region routing:


# Create health check for primary region
aws route53 create-health-check \
    --caller-reference $(date +%s) \
    --health-check-config "Port=443,Type=HTTPS,ResourcePath=/prod/generate,FullyQualifiedDomainName=YOUR_API_ID.execute-api.us-east-1.amazonaws.com,RequestInterval=30,FailureThreshold=3"

# Create hosted zone (if you don't have one)
aws route53 create-hosted-zone \
    --name yourdomain.com \
    --caller-reference $(date +%s)

# Create DNS records with failover routing policy
aws route53 change-resource-record-sets \
    --hosted-zone-id YOUR_HOSTED_ZONE_ID \
    --change-batch '{
        "Changes": [
            {
                "Action": "CREATE",
                "ResourceRecordSet": {
                    "Name": "ai-assistant.yourdomain.com",
                    "Type": "A",
                    "SetIdentifier": "Primary",
                    "Failover": "PRIMARY",
                    "AliasTarget": {
                        "HostedZoneId": "Z1UJRXOUMOOFQ8",
                        "DNSName": "YOUR_API_ID.execute-api.us-east-1.amazonaws.com",
                        "EvaluateTargetHealth": true
                    },
                    "HealthCheckId": "YOUR_HEALTH_CHECK_ID"
                }
            },
            {
                "Action": "CREATE",
                "ResourceRecordSet": {
                    "Name": "ai-assistant.yourdomain.com",
                    "Type": "A",
                    "SetIdentifier": "Secondary",
                    "Failover": "SECONDARY",
                    "AliasTarget": {
                        "HostedZoneId": "Z2OJLYMUO9EFXC",
                        "DNSName": "YOUR_API_ID.execute-api.us-west-2.amazonaws.com",
                        "EvaluateTargetHealth": true
                    }
                }
            }
        ]
    }'
Part 4: Model customization and lifecycle management

Step 1: Fine-tune a model with SageMaker

Prepare a fine-tuning dataset:


import pandas as pd
import json

# Create a financial Q&A dataset
data = [
    {"question": "What is a 401(k)?", "answer": "A 401(k) is a tax-advantaged retirement savings plan offered by employers."},
    {"question": "How does compound interest work?", "answer": "Compound interest is when you earn interest on both the money you've saved and the interest you earn."},
    # Add more examples...
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV for SageMaker training
df.to_csv("financial_qa_dataset.csv", index=False)

# Create SageMaker training script
with open("train.py", "w") as f:
    f.write("""
import argparse
import os
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load dataset
    data_path = os.path.join(args.training_dir, "financial_qa_dataset.csv")
    df = pd.read_csv(data_path)
    
    # Prepare dataset
    def format_instruction(row):
        return f"Question: {row['question']}\\nAnswer: {row['answer']}"
    
    df["text"] = df.apply(format_instruction, axis=1)
    dataset = Dataset.from_pandas(df[["text"]])
    
    # Load model and tokenizer
    model_name = "distilgpt2"  # Use a smaller model for example purposes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples
Share your work with the AWS community

When you share your bonus assignments, the AWS Exam Prep team and others will be able to endorse you on LinkedIn for the skills you demonstrated in this project and in your preparations for your certification exam.

Write a short blog post explaining your learnings and your approach to the challenge.

Link to your project on GitHub.

Tag #awsexamprep, and the AWS Exam Prep team will review it.

Disclaimer:
This project is for educational purposes only and might lead to AWS service charges.
You assume any risk associated with this project. You are responsible for any costs associated with your AWS account.