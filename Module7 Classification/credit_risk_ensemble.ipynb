{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Credit Risk Ensemble Techniques"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from collections import Counter\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import balanced_accuracy_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from imblearn.metrics import classification_report_imbalanced"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Read the CSV and Perform Basic Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_size</th>\n",
       "      <th>interest_rate</th>\n",
       "      <th>homeowner</th>\n",
       "      <th>borrower_income</th>\n",
       "      <th>debt_to_income</th>\n",
       "      <th>num_of_accounts</th>\n",
       "      <th>derogatory_marks</th>\n",
       "      <th>total_debt</th>\n",
       "      <th>loan_status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10700.0</td>\n",
       "      <td>7.672</td>\n",
       "      <td>1</td>\n",
       "      <td>52800</td>\n",
       "      <td>0.431818</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>22800</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8400.0</td>\n",
       "      <td>6.692</td>\n",
       "      <td>1</td>\n",
       "      <td>43600</td>\n",
       "      <td>0.311927</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>13600</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>9000.0</td>\n",
       "      <td>6.963</td>\n",
       "      <td>2</td>\n",
       "      <td>46100</td>\n",
       "      <td>0.349241</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>16100</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10700.0</td>\n",
       "      <td>7.664</td>\n",
       "      <td>1</td>\n",
       "      <td>52700</td>\n",
       "      <td>0.430740</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>22700</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>10800.0</td>\n",
       "      <td>7.698</td>\n",
       "      <td>0</td>\n",
       "      <td>53000</td>\n",
       "      <td>0.433962</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>23000</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   loan_size  interest_rate  homeowner  borrower_income  debt_to_income  \\\n",
       "0    10700.0          7.672          1            52800        0.431818   \n",
       "1     8400.0          6.692          1            43600        0.311927   \n",
       "2     9000.0          6.963          2            46100        0.349241   \n",
       "3    10700.0          7.664          1            52700        0.430740   \n",
       "4    10800.0          7.698          0            53000        0.433962   \n",
       "\n",
       "   num_of_accounts  derogatory_marks  total_debt loan_status  \n",
       "0                5                 1       22800    low_risk  \n",
       "1                3                 0       13600    low_risk  \n",
       "2                3                 0       16100    low_risk  \n",
       "3                5                 1       22700    low_risk  \n",
       "4                5                 1       23000    low_risk  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the data\n",
    "file_path = Path('Resources/lending_data.csv')\n",
    "df = pd.read_csv(file_path)\n",
    "#df.isnull().sum()\n",
    "le = LabelEncoder()\n",
    "le.fit(df[\"homeowner\"])\n",
    "df[\"homeowner\"] = le.transform(df[\"homeowner\"])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Split the Data into Training and Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create our features\n",
    "X = df.drop(columns=\"loan_status\")\n",
    "\n",
    "# Create our target\n",
    "y = df[\"loan_status\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_size</th>\n",
       "      <th>interest_rate</th>\n",
       "      <th>homeowner</th>\n",
       "      <th>borrower_income</th>\n",
       "      <th>debt_to_income</th>\n",
       "      <th>num_of_accounts</th>\n",
       "      <th>derogatory_marks</th>\n",
       "      <th>total_debt</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>77536.000000</td>\n",
       "      <td>77536.000000</td>\n",
       "      <td>77536.000000</td>\n",
       "      <td>77536.000000</td>\n",
       "      <td>77536.000000</td>\n",
       "      <td>77536.000000</td>\n",
       "      <td>77536.000000</td>\n",
       "      <td>77536.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>9805.562577</td>\n",
       "      <td>7.292333</td>\n",
       "      <td>0.606144</td>\n",
       "      <td>49221.949804</td>\n",
       "      <td>0.377318</td>\n",
       "      <td>3.826610</td>\n",
       "      <td>0.392308</td>\n",
       "      <td>19221.949804</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2093.223153</td>\n",
       "      <td>0.889495</td>\n",
       "      <td>0.667811</td>\n",
       "      <td>8371.635077</td>\n",
       "      <td>0.081519</td>\n",
       "      <td>1.904426</td>\n",
       "      <td>0.582086</td>\n",
       "      <td>8371.635077</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5.250000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>30000.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>8700.000000</td>\n",
       "      <td>6.825000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>44800.000000</td>\n",
       "      <td>0.330357</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14800.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>9500.000000</td>\n",
       "      <td>7.172000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>48100.000000</td>\n",
       "      <td>0.376299</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>18100.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>10400.000000</td>\n",
       "      <td>7.528000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>51400.000000</td>\n",
       "      <td>0.416342</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>21400.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>23800.000000</td>\n",
       "      <td>13.235000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>105200.000000</td>\n",
       "      <td>0.714829</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>75200.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          loan_size  interest_rate     homeowner  borrower_income  \\\n",
       "count  77536.000000   77536.000000  77536.000000     77536.000000   \n",
       "mean    9805.562577       7.292333      0.606144     49221.949804   \n",
       "std     2093.223153       0.889495      0.667811      8371.635077   \n",
       "min     5000.000000       5.250000      0.000000     30000.000000   \n",
       "25%     8700.000000       6.825000      0.000000     44800.000000   \n",
       "50%     9500.000000       7.172000      1.000000     48100.000000   \n",
       "75%    10400.000000       7.528000      1.000000     51400.000000   \n",
       "max    23800.000000      13.235000      2.000000    105200.000000   \n",
       "\n",
       "       debt_to_income  num_of_accounts  derogatory_marks    total_debt  \n",
       "count    77536.000000     77536.000000      77536.000000  77536.000000  \n",
       "mean         0.377318         3.826610          0.392308  19221.949804  \n",
       "std          0.081519         1.904426          0.582086   8371.635077  \n",
       "min          0.000000         0.000000          0.000000      0.000000  \n",
       "25%          0.330357         3.000000          0.000000  14800.000000  \n",
       "50%          0.376299         4.000000          0.000000  18100.000000  \n",
       "75%          0.416342         4.000000          1.000000  21400.000000  \n",
       "max          0.714829        16.000000          3.000000  75200.000000  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "low_risk     75036\n",
       "high_risk     2500\n",
       "Name: loan_status, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the balance of our target values\n",
    "y.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(58152, 8)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Split the X and y into X_train, X_test, y_train, y_test\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, \n",
    "                                                    y, \n",
    "                                                    random_state=1, \n",
    "                                                    stratify=y)\n",
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Scale the X_train data\n",
    "scaler = StandardScaler()\n",
    "X_scaler = scaler.fit(X_train)\n",
    "\n",
    "X_train_scaled = X_scaler.transform(X_train)\n",
    "X_test_scaled = X_scaler.transform(X_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Ensemble Learners\n",
    "\n",
    "In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble classifier . For each algorithm, be sure to complete the folliowing steps:\n",
    "\n",
    "1. Train the model using the training data. \n",
    "2. Calculate the balanced accuracy score from sklearn.metrics.\n",
    "3. Print the confusion matrix from sklearn.metrics.\n",
    "4. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.\n",
    "5. For the Balanced Random Forest Classifier onely, print the feature importance sorted in descending order (most important feature to least important) along with the feature score\n",
    "\n",
    "Note: Use a random state of 1 for each algorithm to ensure consistency between tests"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Balanced Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "BalancedRandomForestClassifier(n_estimators=1000, random_state=1)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Resample the training data with the BalancedRandomForestClassifier\n",
    "from imblearn.ensemble import BalancedRandomForestClassifier\n",
    "brf = BalancedRandomForestClassifier(n_estimators=1000, random_state=1)\n",
    "brf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9937351884428807\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score\n",
    "y_pred_rf = brf.predict(X_test)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred_rf)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predicted High RIsk</th>\n",
       "      <th>Predicted Low Risk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Actual High Risk</th>\n",
       "      <td>622</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Low Risk</th>\n",
       "      <td>145</td>\n",
       "      <td>18614</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  622                   3\n",
       "Actual Low Risk                   145               18614"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix\n",
    "cm = confusion_matrix(y_test, y_pred_rf)\n",
    "cm_df = pd.DataFrame(\n",
    "    cm, index=[\"Actual High Risk\", \"Actual Low Risk\"],\n",
    "    columns=[\"Predicted High RIsk\", \"Predicted Low Risk\"])\n",
    "\n",
    "# Displaying results\n",
    "display(cm_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   pre       rec       spe        f1       geo       iba       sup\n",
      "\n",
      "  high_risk    0.81095   0.99520   0.99227   0.89368   0.99373   0.98780       625\n",
      "   low_risk    0.99984   0.99227   0.99520   0.99604   0.99373   0.98722     18759\n",
      "\n",
      "avg / total    0.99375   0.99236   0.99511   0.99274   0.99373   0.98724     19384\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Print the imbalanced classification report\n",
    "print(classification_report_imbalanced(y_test, y_pred_rf, digits = 5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(0.18703526235640708, 'total_debt'),\n",
       " (0.17254127613882989, 'borrower_income'),\n",
       " (0.16565598628726366, 'debt_to_income'),\n",
       " (0.1624914600616813, 'interest_rate'),\n",
       " (0.15613786418656678, 'loan_size'),\n",
       " (0.12193867705419299, 'num_of_accounts'),\n",
       " (0.03158406206866748, 'derogatory_marks'),\n",
       " (0.0026154118463908595, 'homeowner')]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# List the features sorted in descending order by feature importance\n",
    "importances = brf.feature_importances_\n",
    "importances_sorted = sorted(zip(brf.feature_importances_, X.columns), reverse=True)\n",
    "importances_sorted[:8]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "BalancedRandomForestClassifier(n_estimators=1000, random_state=1)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fit the scaled data\n",
    "brf.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9937351884428807\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score for scaled data\n",
    "y_pred_rf_scaled = brf.predict(X_test_scaled)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred_rf_scaled)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predicted High RIsk</th>\n",
       "      <th>Predicted Low Risk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Actual High Risk</th>\n",
       "      <td>622</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Low Risk</th>\n",
       "      <td>145</td>\n",
       "      <td>18614</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  622                   3\n",
       "Actual Low Risk                   145               18614"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix for scaled data\n",
    "cm_scaled = confusion_matrix(y_test, y_pred_rf_scaled)\n",
    "cm_df_scaled = pd.DataFrame(\n",
    "    cm_scaled, index=[\"Actual High Risk\", \"Actual Low Risk\"],\n",
    "    columns=[\"Predicted High RIsk\", \"Predicted Low Risk\"])\n",
    "\n",
    "# Displaying results\n",
    "display(cm_df_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Easy Ensemble Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "EasyEnsembleClassifier(n_estimators=1000, random_state=1)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train the EasyEnsembleClassifier\n",
    "from imblearn.ensemble import EasyEnsembleClassifier\n",
    "eec = EasyEnsembleClassifier(n_estimators=1000, random_state=1)\n",
    "eec.fit(X_train, y_train) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9944548430086891\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score\n",
    "y_pred_eec = eec.predict(X_test)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred_eec)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predicted High RIsk</th>\n",
       "      <th>Predicted Low Risk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Actual High Risk</th>\n",
       "      <td>622</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Low Risk</th>\n",
       "      <td>118</td>\n",
       "      <td>18641</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  622                   3\n",
       "Actual Low Risk                   118               18641"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix\n",
    "cm2 = confusion_matrix(y_test, y_pred_eec)\n",
    "cm2_df = pd.DataFrame(\n",
    "    cm2, index=[\"Actual High Risk\", \"Actual Low Risk\"],\n",
    "    columns=[\"Predicted High RIsk\", \"Predicted Low Risk\"])\n",
    "\n",
    "# Displaying results\n",
    "display(cm2_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   pre       rec       spe        f1       geo       iba       sup\n",
      "\n",
      "  high_risk    0.84054   0.99520   0.99371   0.91136   0.99445   0.98909       625\n",
      "   low_risk    0.99984   0.99371   0.99520   0.99676   0.99445   0.98879     18759\n",
      "\n",
      "avg / total    0.99470   0.99376   0.99515   0.99401   0.99445   0.98880     19384\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Print the imbalanced classification report\n",
    "print(classification_report_imbalanced(y_test, y_pred_eec, digits = 5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "EasyEnsembleClassifier(n_estimators=1000, random_state=1)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fit the scaled data\n",
    "eec.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9944548430086891\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score for scaled data\n",
    "y_pred_eec_scaled = eec.predict(X_test_scaled)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred_eec_scaled)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predicted High RIsk</th>\n",
       "      <th>Predicted Low Risk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Actual High Risk</th>\n",
       "      <td>622</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Low Risk</th>\n",
       "      <td>118</td>\n",
       "      <td>18641</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  622                   3\n",
       "Actual Low Risk                   118               18641"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix for scaled data\n",
    "cm2_scaled = confusion_matrix(y_test, y_pred_eec_scaled)\n",
    "cm2_df_scaled = pd.DataFrame(\n",
    "    cm2_scaled, index=[\"Actual High Risk\", \"Actual Low Risk\"],\n",
    "    columns=[\"Predicted High RIsk\", \"Predicted Low Risk\"])\n",
    "\n",
    "# Displaying results\n",
    "display(cm2_df_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scaling the data does not make any difference to outcomes.\n",
    "\n",
    "#The Easy Ensemble has the better balanced accuracy score.\n",
    "\n",
    "# For High Risk Category, the models are equal on recall score.  For the Low Risk Category, the Easy Ensemble has a better score, so in total\n",
    "# it has a better recall score.\n",
    "\n",
    "#The Easy Ensemble has a better geometric mean.\n",
    "\n",
    "# The top three features are 1. total debt;  2.  borrower income; and 3) debt to income.   The third feature is really linked to #1 and #2.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
