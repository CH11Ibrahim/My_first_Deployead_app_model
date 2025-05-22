import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Define expected feature names based on your training data
expected_features = [
    'id', 'Time', 'Is_CH', 'ADV_S', 'ADV_R', 'JOIN_R', 'SCH_S', 'Rank',
    'DATA_S', 'DATA_R', 'Data_Sent_To_BS', 'dist_CH_To_BS', 'Expaned Energy'
]

# Define the proper class mapping (based on the information you provided)
CLASS_MAPPING = {
    0: 'Blackhole',
    1: 'Flooding',
    2: 'Grayhole',
    3: 'Normal',
    4: 'TDMA'
}

st.title('Network Attack Classification')

# Sidebar for data requirements and model selection
with st.sidebar:
    st.header('Options')
    
    # Model selection dropdown
    model_choice = st.selectbox(
    'Select a model for prediction:',
    ('Decision_Tree_Multi_attack_classifier', 'SVM_Multi_attack_classifier', 'xgboost_attack_classifier', 'Random_Forest_attack_classifier')
)
    
    # Add color scheme option
    color_scheme = st.selectbox(
        'Select visualization color scheme:',
        ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds')
    )
    
    # Input data choice
    st.subheader("Input Data")
    input_option = st.radio(
        "Choose data source:",
        ["Upload your own data", "Use test dataset (X_multi_test & y_multi_test)"]
    )
    
    if input_option == "Upload your own data":
        st.caption('To inference the model, you need to upload a dataframe in CSV format with the required columns.')
        with st.expander('Required Columns'):
            for feature in expected_features:
                st.markdown(f" - {feature}")
    else:
        st.caption('Using pre-defined test datasets (X_multi_test & y_multi_test)')
        st.success("✓ Test data ready for evaluation")
        
    st.divider()
    
    # Evaluation metrics section
    if input_option == "Use test dataset (X_multi_test & y_multi_test)":
        st.subheader("Evaluation Metrics")
        metrics_to_show = st.multiselect(
            "Select metrics to display:",
            ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"],
            default=["Accuracy", "F1 Score", "Confusion Matrix"]
        )
        
    st.caption("<p style='text-align:center'>Developed by CHIBM ;)</p>", unsafe_allow_html=True)

# Initialize session state for app flow control
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

# Only show the start button if we're not using the test dataset option
if input_option == "Upload your own data":
    st.button("Let's get started", on_click=clicked, args=[1])
else:
    # If using test dataset, consider it already "clicked"
    st.session_state.clicked[1] = True

if st.session_state.clicked[1]:
    # Different logic based on the selected input option
    if input_option == "Upload your own data":
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        if uploaded_file is not None:
            # Read CSV with proper formatting
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            # Clean column names: strip spaces and standardize
            df.columns = df.columns.str.strip()
            
            st.header('Uploaded Data Sample')
            st.write(df.head())
            
            # Check for missing features
            missing_features = set(expected_features) - set(df.columns)
            if missing_features:
                st.error(f'Missing features in uploaded data: {", ".join(missing_features)}')
            else:
                st.success("All required features are present in the data.")
                process_data = True
        else:
            process_data = False
    else:
        # Use the test dataset
        try:
            # Load X_multi_test and y_multi_test
            X_test = pd.read_csv('multi-attack-models/X_multi_test.csv', low_memory=False)
            y_test = pd.read_csv('multi-attack-models/y_multi_test.csv', low_memory=False)
            
            # Clean column names
            X_test.columns = X_test.columns.str.strip()
            
            # Display sample of test data
            st.header('Test Data Sample')
            st.write(X_test.head())
            
            # Extract the target variable name (assuming it's the only column in y_test)
            target_col = y_test.columns[0]
            
            # Display target distribution
            st.subheader("Target Distribution in Test Data")
            y_counts = y_test[target_col].value_counts()
            
            # Map numeric target values to class names
            y_counts.index = [CLASS_MAPPING[idx] if idx in CLASS_MAPPING else f"Unknown-{idx}" for idx in y_counts.index]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_counts.plot.bar(ax=ax, color=plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(y_counts))))
            ax.set_title('Actual Class Distribution in Test Data')
            ax.set_xlabel('Attack Type')
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Use X_test as our df for processing
            df = X_test
            
            # Also store the ground truth for evaluation
            actual_classes = [CLASS_MAPPING[idx] if idx in CLASS_MAPPING else f"Unknown-{idx}" for idx in y_test[target_col]]
            
            # Set process_data flag
            process_data = True
            st.success("Test data loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading test data: {str(e)}")
            st.error("Please make sure X_multi_test.csv and y_multi_test.csv are in the same directory as your app.")
            process_data = False
            
    # Process data if it's valid (either uploaded or test data)
    if 'process_data' in locals() and process_data:    
        # Ensure columns are in the correct order and format for the model
        try:
            # Load the selected model
            model_path = f'multi-attack-models/{model_choice}.joblib'
            model = joblib.load(model_path)
            
            # Use the predefined class mapping instead of loading the label encoder
            class_names = [CLASS_MAPPING[i] for i in range(5)]  # Use our defined mapping
            st.success(f"Using class mapping: {CLASS_MAPPING}")
                            
            # Create a new DataFrame with only the expected features
            filtered_df = pd.DataFrame()
            for feature in expected_features:
                if feature in df.columns:
                    filtered_df[feature] = df[feature]
                else:
                    st.warning(f"Feature '{feature}' not found in data, filling with zeros")
                    filtered_df[feature] = np.zeros(len(df))
            
            # Fix any possible whitespace issues in column names
            filtered_df.columns = [col.strip() for col in filtered_df.columns]
            
            # Show the final features being used for prediction
            st.write(f"Selected Features: {filtered_df.columns.tolist()}")
            
            # Apply the model directly to avoid feature name issues
            X = filtered_df.values
            
            # Make predictions using the raw values
            pred_proba = model.predict_proba(X)
            pred_class_idx = np.argmax(pred_proba, axis=1)
            
            # Convert indices to class names using our mapping
            pred_class_names = [CLASS_MAPPING[idx] if idx in CLASS_MAPPING else f"Unknown-{idx}" for idx in pred_class_idx]
            
            # Create prediction DataFrame with both probabilities and predicted class
            pred_df = pd.DataFrame(pred_proba, columns=class_names)
            pred_df['Predicted_Class'] = pred_class_names
            
            # Add original data for reference
            result_df = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            
            # Show summary of classifications with colorful visualization
            st.header('Prediction Results')
            
            # Classification summary with chart
            st.subheader("Classification Summary")
            class_counts = pd.Series(pred_class_names).value_counts()
            
            # Create a pie chart of attack distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Pie chart
            class_counts.plot.pie(autopct='%1.1f%%', ax=ax1, cmap=color_scheme)
            ax1.set_title('Distribution of Predicted Attacks')
            ax1.set_ylabel('')
            
            # Bar chart
            class_counts.plot.bar(ax=ax2, color=plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(class_counts))))
            ax2.set_title('Count of Predicted Attacks')
            ax2.set_xlabel('Attack Type')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Model evaluation if using test data
            if input_option == "Use test dataset (X_multi_test & y_multi_test)" and 'actual_classes' in locals():
                st.header("Model Evaluation on Test Data")
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                # Calculate metrics
                accuracy = accuracy_score(actual_classes, pred_class_names)
                
                # Display selected metrics
                metrics_col1, metrics_col2 = st.columns(2)
                
                if "Accuracy" in metrics_to_show:
                    metrics_col1.metric("Accuracy", f"{accuracy:.4f}")
                
                if "Precision" in metrics_to_show:
                    precision = precision_score(actual_classes, pred_class_names, average='weighted')
                    metrics_col1.metric("Precision (weighted)", f"{precision:.4f}")
                
                if "Recall" in metrics_to_show:
                    recall = recall_score(actual_classes, pred_class_names, average='weighted')
                    metrics_col2.metric("Recall (weighted)", f"{recall:.4f}")
                
                if "F1 Score" in metrics_to_show:
                    f1 = f1_score(actual_classes, pred_class_names, average='weighted')
                    metrics_col2.metric("F1 Score (weighted)", f"{f1:.4f}")
                
                if "Confusion Matrix" in metrics_to_show:
                    st.subheader("Confusion Matrix")
                    
                    # Get unique classes from both actual and predicted
                    unique_classes = sorted(list(set(actual_classes + pred_class_names)))
                    
                    # Calculate confusion matrix
                    cm = confusion_matrix(actual_classes, pred_class_names, labels=unique_classes)
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap=color_scheme, 
                              xticklabels=unique_classes, yticklabels=unique_classes)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Show comparison of actual vs predicted
                st.subheader("Actual vs. Predicted Classes")
                comparison_df = pd.DataFrame({
                    'Actual': actual_classes,
                    'Predicted': pred_class_names,
                    'Match': [a == p for a, p in zip(actual_classes, pred_class_names)]
                })
                
                # Display comparison dataframe with highlighting
                def highlight_matches(val):
                    return 'background-color: lightgreen' if val else 'background-color: lightcoral'
                
                styled_comparison = comparison_df.head(20).style.applymap(highlight_matches, subset=['Match'])
                st.write(styled_comparison)
                
                # Mismatch analysis
                mismatches = comparison_df[~comparison_df['Match']]
                if len(mismatches) > 0:
                    st.subheader("Misclassification Analysis")
                    
                    # Count and chart the types of misclassifications
                    mismatch_counts = mismatches.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
                    mismatch_counts = mismatch_counts.sort_values('Count', ascending=False)
                    
                    st.write("Top misclassification patterns:")
                    st.write(mismatch_counts.head(10))
                    
                    # Create bar chart of top misclassification patterns
                    if len(mismatch_counts) > 0:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        top_mismatches = mismatch_counts.head(10)
                        labels = [f"{a} → {p}" for a, p in zip(top_mismatches['Actual'], top_mismatches['Predicted'])]
                        plt.bar(labels, top_mismatches['Count'], 
                               color=plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(top_mismatches))))
                        plt.xticks(rotation=45, ha='right')
                        plt.title('Top Misclassification Patterns')
                        plt.xlabel('Actual → Predicted')
                        plt.ylabel('Count')
                        plt.tight_layout()
                        st.pyplot(fig)
                
            # Data summary
            st.subheader("Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(df))
                st.metric("Normal Traffic", len(pred_df[pred_df['Predicted_Class'] == 'Normal']))
            with col2:
                attack_count = len(pred_df[pred_df['Predicted_Class'] != 'Normal'])
                st.metric("Attack Traffic", attack_count)
                if len(df) > 0:
                    st.metric("Attack Percentage", f"{attack_count/len(df)*100:.2f}%")
            
            # Show prediction probabilities with highlighted max values
            st.subheader("Prediction Details")
            
            # Function to highlight the maximum probability
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: yellow' if v else '' for v in is_max]
            
            # Display styled dataframe with highlighted max probabilities
            styled_df = pred_df.head(10).style.apply(highlight_max, axis=1, subset=class_names)
            st.write(styled_df)
            
            # Show full results with original data
            with st.expander("Full Results (with original data)"):
                st.dataframe(result_df)
            
            # Confidence analysis
            st.subheader("Prediction Confidence Analysis")
            
            # Extract max probability for each prediction as confidence score
            confidence_scores = pred_proba.max(axis=1)
            
            # Create figure for confidence distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(confidence_scores, bins=20, kde=True, ax=ax, color=plt.cm.get_cmap(color_scheme)(0.6))
            ax.set_title('Distribution of Prediction Confidence')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
            # Show confidence by class
            fig, ax = plt.subplots(figsize=(12, 6))
            confidence_by_class = pd.DataFrame({
                'Class': pred_class_names,
                'Confidence': confidence_scores
            })
            
            sns.boxplot(x='Class', y='Confidence', data=confidence_by_class, ax=ax, 
                       palette=plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(class_counts))))
            ax.set_title('Prediction Confidence by Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Prepare for download
            pred_csv = pred_df.to_csv(index=False).encode('utf-8')
            full_csv = result_df.to_csv(index=False).encode('utf-8')
            
            # Download buttons
            st.subheader("Download Results")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button('Download Predictions Only',
                                 pred_csv,
                                 'predictions.csv',
                                 'text/csv',
                                 key='download-pred-csv')
            with col2:
                st.download_button('Download Full Results',
                                 full_csv,
                                 'full_results.csv',
                                 'text/csv',
                                 key='download-full-csv')
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Debug info:")
            st.write(f"DataFrame columns: {df.columns.tolist()}")
            st.write(f"Expected features: {expected_features}")
            st.write("If the error persists, please check the paths to your model files and ensure they are accessible.")