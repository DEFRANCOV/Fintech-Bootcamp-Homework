{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Credit Risk Resampling Techniques"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
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
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from collections import Counter\n",
    "from sklearn.preprocessing import LabelEncoder"
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
   "execution_count": 48,
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
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>10100.0</td>\n",
       "      <td>7.438</td>\n",
       "      <td>0</td>\n",
       "      <td>50600</td>\n",
       "      <td>0.407115</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>20600</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>10300.0</td>\n",
       "      <td>7.490</td>\n",
       "      <td>0</td>\n",
       "      <td>51100</td>\n",
       "      <td>0.412916</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>21100</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8800.0</td>\n",
       "      <td>6.857</td>\n",
       "      <td>0</td>\n",
       "      <td>45100</td>\n",
       "      <td>0.334812</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>15100</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9300.0</td>\n",
       "      <td>7.096</td>\n",
       "      <td>1</td>\n",
       "      <td>47400</td>\n",
       "      <td>0.367089</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>17400</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>9700.0</td>\n",
       "      <td>7.248</td>\n",
       "      <td>2</td>\n",
       "      <td>48800</td>\n",
       "      <td>0.385246</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>18800</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>9300.0</td>\n",
       "      <td>7.085</td>\n",
       "      <td>1</td>\n",
       "      <td>47300</td>\n",
       "      <td>0.365751</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>17300</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>9200.0</td>\n",
       "      <td>7.015</td>\n",
       "      <td>1</td>\n",
       "      <td>46600</td>\n",
       "      <td>0.356223</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>16600</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>9200.0</td>\n",
       "      <td>7.043</td>\n",
       "      <td>1</td>\n",
       "      <td>46900</td>\n",
       "      <td>0.360341</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>16900</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>9400.0</td>\n",
       "      <td>7.101</td>\n",
       "      <td>0</td>\n",
       "      <td>47400</td>\n",
       "      <td>0.367089</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>17400</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>8400.0</td>\n",
       "      <td>6.703</td>\n",
       "      <td>0</td>\n",
       "      <td>43700</td>\n",
       "      <td>0.313501</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>13700</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>9400.0</td>\n",
       "      <td>7.136</td>\n",
       "      <td>0</td>\n",
       "      <td>47700</td>\n",
       "      <td>0.371069</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>17700</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>8900.0</td>\n",
       "      <td>6.897</td>\n",
       "      <td>1</td>\n",
       "      <td>45500</td>\n",
       "      <td>0.340659</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>15500</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>8300.0</td>\n",
       "      <td>6.648</td>\n",
       "      <td>0</td>\n",
       "      <td>43200</td>\n",
       "      <td>0.305556</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>13200</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>8000.0</td>\n",
       "      <td>6.539</td>\n",
       "      <td>0</td>\n",
       "      <td>42100</td>\n",
       "      <td>0.287411</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12100</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>10200.0</td>\n",
       "      <td>7.460</td>\n",
       "      <td>2</td>\n",
       "      <td>50800</td>\n",
       "      <td>0.409449</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>20800</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>5200.0</td>\n",
       "      <td>5.326</td>\n",
       "      <td>0</td>\n",
       "      <td>30700</td>\n",
       "      <td>0.022801</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>700</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>9300.0</td>\n",
       "      <td>7.068</td>\n",
       "      <td>1</td>\n",
       "      <td>47100</td>\n",
       "      <td>0.363057</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>17100</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>7300.0</td>\n",
       "      <td>6.225</td>\n",
       "      <td>1</td>\n",
       "      <td>39200</td>\n",
       "      <td>0.234694</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>9200</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>9000.0</td>\n",
       "      <td>6.943</td>\n",
       "      <td>1</td>\n",
       "      <td>45900</td>\n",
       "      <td>0.346405</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>15900</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>8700.0</td>\n",
       "      <td>6.813</td>\n",
       "      <td>1</td>\n",
       "      <td>44700</td>\n",
       "      <td>0.328859</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>14700</td>\n",
       "      <td>low_risk</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    loan_size  interest_rate  homeowner  borrower_income  debt_to_income  \\\n",
       "0     10700.0          7.672          1            52800        0.431818   \n",
       "1      8400.0          6.692          1            43600        0.311927   \n",
       "2      9000.0          6.963          2            46100        0.349241   \n",
       "3     10700.0          7.664          1            52700        0.430740   \n",
       "4     10800.0          7.698          0            53000        0.433962   \n",
       "5     10100.0          7.438          0            50600        0.407115   \n",
       "6     10300.0          7.490          0            51100        0.412916   \n",
       "7      8800.0          6.857          0            45100        0.334812   \n",
       "8      9300.0          7.096          1            47400        0.367089   \n",
       "9      9700.0          7.248          2            48800        0.385246   \n",
       "10     9300.0          7.085          1            47300        0.365751   \n",
       "11     9200.0          7.015          1            46600        0.356223   \n",
       "12     9200.0          7.043          1            46900        0.360341   \n",
       "13     9400.0          7.101          0            47400        0.367089   \n",
       "14     8400.0          6.703          0            43700        0.313501   \n",
       "15     9400.0          7.136          0            47700        0.371069   \n",
       "16     8900.0          6.897          1            45500        0.340659   \n",
       "17     8300.0          6.648          0            43200        0.305556   \n",
       "18     8000.0          6.539          0            42100        0.287411   \n",
       "19    10200.0          7.460          2            50800        0.409449   \n",
       "20     5200.0          5.326          0            30700        0.022801   \n",
       "21     9300.0          7.068          1            47100        0.363057   \n",
       "22     7300.0          6.225          1            39200        0.234694   \n",
       "23     9000.0          6.943          1            45900        0.346405   \n",
       "24     8700.0          6.813          1            44700        0.328859   \n",
       "\n",
       "    num_of_accounts  derogatory_marks  total_debt loan_status  \n",
       "0                 5                 1       22800    low_risk  \n",
       "1                 3                 0       13600    low_risk  \n",
       "2                 3                 0       16100    low_risk  \n",
       "3                 5                 1       22700    low_risk  \n",
       "4                 5                 1       23000    low_risk  \n",
       "5                 4                 1       20600    low_risk  \n",
       "6                 4                 1       21100    low_risk  \n",
       "7                 3                 0       15100    low_risk  \n",
       "8                 3                 0       17400    low_risk  \n",
       "9                 4                 0       18800    low_risk  \n",
       "10                3                 0       17300    low_risk  \n",
       "11                3                 0       16600    low_risk  \n",
       "12                3                 0       16900    low_risk  \n",
       "13                3                 0       17400    low_risk  \n",
       "14                3                 0       13700    low_risk  \n",
       "15                3                 0       17700    low_risk  \n",
       "16                3                 0       15500    low_risk  \n",
       "17                2                 0       13200    low_risk  \n",
       "18                2                 0       12100    low_risk  \n",
       "19                4                 1       20800    low_risk  \n",
       "20                0                 0         700    low_risk  \n",
       "21                3                 0       17100    low_risk  \n",
       "22                2                 0        9200    low_risk  \n",
       "23                3                 0       15900    low_risk  \n",
       "24                3                 0       14700    low_risk  "
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the data\n",
    "file_path = Path('Resources/lending_data.csv')\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "#NO NULLS SEEN      df.isnull().sum()\n",
    "\n",
    "#There were duplicates but can't apply because there isn't any unique identifier that is duplicated      df.duplicated()  \n",
    "\n",
    "# encode the homeowner column\n",
    "le = LabelEncoder()\n",
    "le.fit(df[\"homeowner\"])\n",
    "df[\"homeowner\"] = le.transform(df[\"homeowner\"])\n",
    "\n",
    "\n",
    "df.head(25)"
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
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create our features\n",
    "X = df.drop(columns=\"loan_status\")\n",
    "\n",
    "# Create our target\n",
    "y = df[\"loan_status\"]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.describe()\n",
    "#y.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
     "execution_count": 6,
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(58152, 8)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create X_train, X_test, y_train, y_test\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, \n",
    "                                                    y, \n",
    "                                                    random_state=1, \n",
    "                                                    stratify=y)\n",
    "X_train.shape\n",
    "#X_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simple Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(random_state=1)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "model = LogisticRegression(solver='lbfgs', random_state=1)\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9543211898288821\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score\n",
    "from sklearn.metrics import balanced_accuracy_score\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "#balanced_accuracy_score(y_test, y_pred)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred)}\")\n",
    "#print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred)*100}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
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
       "      <td>571</td>\n",
       "      <td>54</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Low Risk</th>\n",
       "      <td>93</td>\n",
       "      <td>18666</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  571                  54\n",
       "Actual Low Risk                    93               18666"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "#confusion_matrix(y_test, y_pred)\n",
    "#confusion_matrix(y_pred, y_test)\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
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
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   pre       rec       spe        f1       geo       iba       sup\n",
      "\n",
      "  high_risk    0.85994   0.91360   0.99504   0.88596   0.95345   0.90167       625\n",
      "   low_risk    0.99712   0.99504   0.91360   0.99608   0.95345   0.91647     18759\n",
      "\n",
      "avg / total    0.99269   0.99242   0.91623   0.99253   0.95345   0.91600     19384\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Print the imbalanced classification report\n",
    "from imblearn.metrics import classification_report_imbalanced\n",
    "print(classification_report_imbalanced(y_test, y_pred, digits = 5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Oversampling\n",
    "\n",
    "In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm. For each algorithm, be sure to complete the folliowing steps:\n",
    "\n",
    "1. View the count of the target classes using `Counter` from the collections library. \n",
    "3. Use the resampled data to train a logistic regression model.\n",
    "3. Calculate the balanced accuracy score from sklearn.metrics.\n",
    "4. Print the confusion matrix from sklearn.metrics.\n",
    "5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.\n",
    "\n",
    "Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Naive Random Oversampling"
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
       "Counter({'low_risk': 56277, 'high_risk': 56277})"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Resample the training data with the RandomOversampler\n",
    "from imblearn.over_sampling import RandomOverSampler\n",
    "\n",
    "ros = RandomOverSampler(random_state=1)\n",
    "X_resampled, y_resampled = ros.fit_resample(X_train, y_train)\n",
    "Counter(y_resampled)"
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
       "LogisticRegression(random_state=1)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train the Logistic Regression model using the resampled data\n",
    "model2 = LogisticRegression(solver='lbfgs', random_state=1)\n",
    "model2.fit(X_resampled, y_resampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9948013433551894\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score\n",
    "y_pred2 = model2.predict(X_test)\n",
    "\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred2)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
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
       "      <td>105</td>\n",
       "      <td>18654</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  622                   3\n",
       "Actual Low Risk                   105               18654"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix\n",
    "cm2 = confusion_matrix(y_test, y_pred2)\n",
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
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   pre       rec       spe        f1       geo       iba       sup\n",
      "\n",
      "  high_risk    0.85557   0.99520   0.99440   0.92012   0.99480   0.98971       625\n",
      "   low_risk    0.99984   0.99440   0.99520   0.99711   0.99480   0.98955     18759\n",
      "\n",
      "avg / total    0.99519   0.99443   0.99517   0.99463   0.99480   0.98956     19384\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Print the imbalanced classification report\n",
    "print(classification_report_imbalanced(y_test, y_pred2, digits = 5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SMOTE Oversampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({'low_risk': 56277, 'high_risk': 56277})"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Resample the training data with SMOTE\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "X_resampled2, y_resampled2 = SMOTE(random_state=1, sampling_strategy=1.0).fit_resample(\n",
    "    X_train, y_train\n",
    ")\n",
    "\n",
    "\n",
    "Counter(y_resampled2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(random_state=1)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train the Logistic Regression model using the resampled data\n",
    "model3 = LogisticRegression(solver='lbfgs', random_state=1)\n",
    "model3.fit(X_resampled2, y_resampled2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9948279972279972\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score\n",
    "y_pred3 = model3.predict(X_test)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred3)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
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
       "      <td>104</td>\n",
       "      <td>18655</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  622                   3\n",
       "Actual Low Risk                   104               18655"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix\n",
    "cm3 = confusion_matrix(y_test, y_pred3)\n",
    "cm3_df = pd.DataFrame(\n",
    "    cm3, index=[\"Actual High Risk\", \"Actual Low Risk\"],\n",
    "    columns=[\"Predicted High RIsk\", \"Predicted Low Risk\"])\n",
    "\n",
    "# Displaying results\n",
    "display(cm3_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   pre       rec       spe        f1       geo       iba       sup\n",
      "\n",
      "  high_risk    0.85675   0.99520   0.99446   0.92080   0.99483   0.98976       625\n",
      "   low_risk    0.99984   0.99446   0.99520   0.99714   0.99483   0.98961     18759\n",
      "\n",
      "avg / total    0.99523   0.99448   0.99518   0.99468   0.99483   0.98961     19384\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Print the imbalanced classification report\n",
    "print(classification_report_imbalanced(y_test, y_pred3, digits = 5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Undersampling\n",
    "\n",
    "In this section, you will test an undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. You will undersample the data using the Cluster Centroids algorithm and complete the folliowing steps:\n",
    "\n",
    "1. View the count of the target classes using `Counter` from the collections library. \n",
    "3. Use the resampled data to train a logistic regression model.\n",
    "3. Calculate the balanced accuracy score from sklearn.metrics.\n",
    "4. Print the confusion matrix from sklearn.metrics.\n",
    "5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.\n",
    "\n",
    "Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({'high_risk': 1875, 'low_risk': 1875})"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Resample the data using the ClusterCentroids resampler\n",
    "from imblearn.under_sampling import ClusterCentroids\n",
    "\n",
    "cc = ClusterCentroids(random_state=1)\n",
    "X_resampled3, y_resampled3 = cc.fit_resample(X_train, y_train)\n",
    "\n",
    "Counter(y_resampled3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(random_state=1)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train the Logistic Regression model using the resampled data\n",
    "model4 = LogisticRegression(solver='lbfgs', random_state=1)\n",
    "model4.fit(X_resampled3, y_resampled3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.9836813049736126\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score\n",
    "y_pred4 = model4.predict(X_test)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred4)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
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
       "      <td>608</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Low Risk</th>\n",
       "      <td>102</td>\n",
       "      <td>18657</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  608                  17\n",
       "Actual Low Risk                   102               18657"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix\n",
    "cm4 = confusion_matrix(y_test, y_pred4)\n",
    "cm4_df = pd.DataFrame(\n",
    "    cm4, index=[\"Actual High Risk\", \"Actual Low Risk\"],\n",
    "    columns=[\"Predicted High RIsk\", \"Predicted Low Risk\"])\n",
    "\n",
    "# Displaying results\n",
    "display(cm4_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   pre       rec       spe        f1       geo       iba       sup\n",
      "\n",
      "  high_risk    0.85634   0.97280   0.99456   0.91086   0.98362   0.96540       625\n",
      "   low_risk    0.99909   0.99456   0.97280   0.99682   0.98362   0.96962     18759\n",
      "\n",
      "avg / total    0.99449   0.99386   0.97350   0.99405   0.98362   0.96948     19384\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Print the imbalanced classification report\n",
    "print(classification_report_imbalanced(y_test, y_pred4, digits = 5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Combination (Over and Under) Sampling\n",
    "\n",
    "In this section, you will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. You will resample the data using the SMOTEENN algorithm and complete the folliowing steps:\n",
    "\n",
    "1. View the count of the target classes using `Counter` from the collections library. \n",
    "3. Use the resampled data to train a logistic regression model.\n",
    "3. Calculate the balanced accuracy score from sklearn.metrics.\n",
    "4. Print the confusion matrix from sklearn.metrics.\n",
    "5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.\n",
    "\n",
    "Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({'high_risk': 74117, 'low_risk': 74577})"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Resample the training data with SMOTEENN\n",
    "from imblearn.combine import SMOTEENN\n",
    "\n",
    "smoat_eenn = SMOTEENN(random_state=1)\n",
    "X_resampled4, y_resampled4 = smoat_eenn.fit_resample(X, y)\n",
    "Counter(y_resampled4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(random_state=1)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train the Logistic Regression model using the resampled data\n",
    "model5 = LogisticRegression(solver='lbfgs', random_state=1)\n",
    "model5.fit(X_resampled4, y_resampled4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The balanced accuracy score is: 0.994748035609574\n"
     ]
    }
   ],
   "source": [
    "# Calculated the balanced accuracy score\n",
    "y_pred5 = model5.predict(X_test)\n",
    "print(f\"The balanced accuracy score is: {balanced_accuracy_score(y_test, y_pred5)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
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
       "      <td>107</td>\n",
       "      <td>18652</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Predicted High RIsk  Predicted Low Risk\n",
       "Actual High Risk                  622                   3\n",
       "Actual Low Risk                   107               18652"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Display the confusion matrix\n",
    "cm5 = confusion_matrix(y_test, y_pred5)\n",
    "cm5_df = pd.DataFrame(\n",
    "    cm5, index=[\"Actual High Risk\", \"Actual Low Risk\"],\n",
    "    columns=[\"Predicted High RIsk\", \"Predicted Low Risk\"])\n",
    "\n",
    "# Displaying results\n",
    "display(cm5_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   pre       rec       spe        f1       geo       iba       sup\n",
      "\n",
      "  high_risk    0.85322   0.99520   0.99430   0.91876   0.99475   0.98961       625\n",
      "   low_risk    0.99984   0.99430   0.99520   0.99706   0.99475   0.98943     18759\n",
      "\n",
      "avg / total    0.99511   0.99433   0.99517   0.99454   0.99475   0.98944     19384\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Print the imbalanced classification report\n",
    "print(classification_report_imbalanced(y_test, y_pred5, digits = 5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Q&A\n",
    "\n",
    "# The SMOTE Oversampling has the best balanced accuracy score, just ever so slightly better than the Random Oversampling and SMOTEENN.\n",
    "\n",
    "# If we were to focus only on the High Risk Category, then SMOTE, Random Oversampling and SMOTEENN are tied for recall risk.  When we analyze the\n",
    "# the Low Risk Category, then SMOTE is the winner ever so slightly.   \n",
    "# Recall for High Risk is the most important measure, but since we're tied here, than overall SMOTE has the best Recall Score by virtue of having \n",
    "# the best Low Risk score.\n",
    "\n",
    "# The SMOTE has the best geometric mean, just beating the Random Oversampling."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:pyvizenv] *",
   "language": "python",
   "name": "conda-env-pyvizenv-py"
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
