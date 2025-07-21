Technical Report on Deep Route Recommendation System Using LSTM

Author

Dhinna Tretarnthip

Supervisor

Professor Tuna Çakar

Date

July 21, 2025

Abstract

This report documents the development, implementation, and evaluation of an LSTM-based deep route recommendation system utilizing GPS trajectory data from the Porto Taxi Dataset integrated with ERA5-Land weather data. It highlights all the stages of data preprocessing, model training, evaluation metrics, encountered challenges, and future improvement possibilities.

Introduction

The primary objective of this Proof of Concept (PoC) project is to explore and demonstrate the capabilities of Long Short-Term Memory (LSTM) neural networks to recommend personalized travel routes using historical GPS trajectories augmented with relevant weather information.

Data Preparation and Preprocessing

Porto Taxi Dataset
	•	Original source: Porto GPS trajectory data.
	•	Spatial grid size: 500m.
	•	Window size: 10 steps (win10).

ERA5-Land Weather Dataset
	•	Included variables: Temperature, Precipitation, Wind Speed, etc.
	•	Temporal resolution: Hourly data aligned spatially and temporally with GPS trajectories.

Data Integration

The datasets were successfully merged to form the final combined dataset (porto_sequences_win10.h5). This integration was achieved through precise spatial-temporal alignment techniques, enriching trajectory data with relevant weather variables.

Model Development

LSTM Architecture
	•	Embedding dimension: 128
	•	LSTM hidden units: 256
	•	Final dense layer: Softmax classifier to predict next cell.

Hyperparameters
	•	Batch size: 2048
	•	Epochs: 7
	•	Device used: MacBook Air M3 (MPS backend)

Training

The training dataset contained approximately 11.8 million sequences, and the training was conducted over 7 epochs.

Evaluation

Dataset Splits

The preprocessed dataset was split into training, validation, and testing subsets as follows:
	•	Training Set:  Approx. 22 million sequences
	•	Validation Set: Approx. 6 million sequences
	•	Test Set (Combined): Approx. 11.8 million sequences
	•	Warm-start Test Subset: Approx. 3.3 million sequences
	•	Cold-start Test Subset: Approx. 8.5 million sequences

Results

Dataset	Top-1 Accuracy	Top-5 Accuracy
Combined	64.53%	95.65%
Warm Set	64.72%	95.67%
Cold Set	64.46%	95.65%

Results visualization saved in eval_route_epoch7.png.

Challenges and Issues Encountered

Technical Challenges
	1.	Checkpoint Loading Issue: Initial mismatch between saved checkpoints and model dimensions required a checkpoint recreation.
	2.	Multiprocessing Data Loader: Encountered pickling issues with h5py objects when multiprocessing was enabled. This was resolved by setting workers to 0.
	3.	Logging Issues: Formatting errors in the logging statements caused runtime exceptions; rectified by correcting the formatting string.
	4.	GPU/Backend Limitations: Encountered issues specific to MPS (Apple Silicon GPU backend), such as unsupported pinned memory warnings and internal errors. These were managed by tuning configurations.

Script Execution Challenges
	•	Encountered errors in parsing Top-1/Top-5 accuracies from subprocess output. This was resolved by refining regex patterns used for parsing.

Discussion and Analysis

The LSTM model demonstrates reasonable prediction accuracy given the large dataset and computational constraints. The integration of weather data provided contextual improvements in predictive modeling, which could be expanded upon further.

Future Work
	•	Conduct rigorous hyperparameter tuning and explore alternative neural network architectures.
	•	Investigate techniques to handle larger datasets and computational resources, including cloud deployments.
	•	Explore real-time deployment possibilities.
	•	Evaluate model robustness with additional external datasets or real-world trials.

Conclusion

This PoC successfully validated the feasibility of LSTM-based route recommendation with integrated weather features. Addressing identified limitations and implementing suggested improvements will further enhance the effectiveness and applicability of the system.
