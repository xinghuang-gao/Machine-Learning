{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Determine the max information gain of entropy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import libraries\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "# read data\n",
    "\n",
    "data = pd.read_csv(\"/Users/xing-huanggao/Desktop/bugs.csv\", header = 0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 24 entries, 0 to 23\n",
      "Data columns (total 3 columns):\n",
      "Species        24 non-null object\n",
      "Color          24 non-null object\n",
      "Length (mm)    24 non-null float64\n",
      "dtypes: float64(1), object(2)\n",
      "memory usage: 704.0+ bytes\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Species', 'Color', 'Length (mm)'], dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
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
       "      <th>Length (mm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>count</td>\n",
       "      <td>24.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>mean</td>\n",
       "      <td>18.070833</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>std</td>\n",
       "      <td>3.584323</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>min</td>\n",
       "      <td>11.600000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>25%</td>\n",
       "      <td>14.975000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>50%</td>\n",
       "      <td>18.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>75%</td>\n",
       "      <td>20.625000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>max</td>\n",
       "      <td>24.800000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Length (mm)\n",
       "count    24.000000\n",
       "mean     18.070833\n",
       "std       3.584323\n",
       "min      11.600000\n",
       "25%      14.975000\n",
       "50%      18.300000\n",
       "75%      20.625000\n",
       "max      24.800000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Lobug    14\n",
       "Mobug    10\n",
       "Name: Species, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.Species.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Blue     10\n",
       "Green     8\n",
       "Brown     6\n",
       "Name: Color, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.Color.value_counts()"
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
       "17"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(data[\"Length (mm)\"]<20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
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
       "      <th>Species</th>\n",
       "      <th>Color</th>\n",
       "      <th>Length (mm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Mobug</td>\n",
       "      <td>Brown</td>\n",
       "      <td>11.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>Mobug</td>\n",
       "      <td>Blue</td>\n",
       "      <td>16.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>Lobug</td>\n",
       "      <td>Blue</td>\n",
       "      <td>15.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>Mobug</td>\n",
       "      <td>Brown</td>\n",
       "      <td>15.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>Mobug</td>\n",
       "      <td>Brown</td>\n",
       "      <td>13.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>13</td>\n",
       "      <td>Lobug</td>\n",
       "      <td>Blue</td>\n",
       "      <td>14.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>19</td>\n",
       "      <td>Mobug</td>\n",
       "      <td>Blue</td>\n",
       "      <td>14.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>21</td>\n",
       "      <td>Lobug</td>\n",
       "      <td>Brown</td>\n",
       "      <td>14.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>23</td>\n",
       "      <td>Mobug</td>\n",
       "      <td>Blue</td>\n",
       "      <td>13.1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Species  Color  Length (mm)\n",
       "0    Mobug  Brown         11.6\n",
       "1    Mobug   Blue         16.3\n",
       "2    Lobug   Blue         15.1\n",
       "6    Mobug  Brown         15.7\n",
       "12   Mobug  Brown         13.8\n",
       "13   Lobug   Blue         14.5\n",
       "19   Mobug   Blue         14.6\n",
       "21   Lobug  Brown         14.1\n",
       "23   Mobug   Blue         13.1"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# count # of mobug and lobug\n",
    "data[data[\"Length (mm)\"] <17]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Lobug    11\n",
       "Mobug     4\n",
       "Name: Species, dtype: int64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data[\"Length (mm)\"] >17].Species.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Mobug    6\n",
       "Lobug    3\n",
       "Name: Species, dtype: int64"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data[\"Length (mm)\"] <17].Species.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = -(3/9*np.log2(3/9) + 6/9 * np.log2(6/9))\n",
    "b = -(11/15 * np.log2(11/15) + 4/15*np.log2(4/15))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.867261401483663"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a*9/24 + b *15/24"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.11260735516748976"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "S_parent - (a*9/24 + b *15/24)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9798687566511528"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def two_group_ent(first, tot):                        \n",
    "    return -(first/tot*np.log2(first/tot) +           \n",
    "             (tot-first)/tot*np.log2((tot-first)/tot))\n",
    "tot_ent = two_group_ent(10, 24)\n",
    "tot_ent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8366407419411672"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "two_group_ent(11,15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.867261401483663"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "g17_ent = 15/24*two_group_ent(11,15) + 9/24*two_group_ent(6,9)\n",
    "\n",
    "g17_ent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Mobug    9\n",
       "Lobug    8\n",
       "Name: Species, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data[\"Length (mm)\"] <20].Species.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Lobug    6\n",
       "Mobug    1\n",
       "Name: Species, dtype: int64"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data[\"Length (mm)\"] >20].Species.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Blue     6\n",
       "Green    6\n",
       "Brown    2\n",
       "Name: Color, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data[\"Species\"] == \"Lobug\"].Color.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXoAAAD4CAYAAADiry33AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3deXxU9b3/8ddnZrLvK1khgbCGsEaksrgjiIBrW7tcbe21+rv+WrV30Xu72PZeb+212p/F2osbVq07KsjiUlELsoU17IQQskPIvq/f3x8ZKIYACUxyZvk8Hw8eTGbOzLwPQ945+Z5zvkeMMSillPJeNqsDKKWUGlha9Eop5eW06JVSystp0SullJfToldKKS/nsDpAT7GxsSYtLc3qGEop5VG2bt16whgT19tjblf0aWlp5OTkWB1DKaU8iogcPdtjOnSjlFJeToteKaW8nBa9Ukp5Obcbo1dKuZf29naKi4tpaWmxOooCAgMDSUlJwc/Pr8/P0aJXSp1TcXExYWFhpKWlISJWx/FpxhgqKyspLi4mPT29z8/ToRul1Dm1tLQQExOjJe8GRISYmJh+/3alRa+UOi8tefdxIZ+FFr1S/dDR2UVXl07trTyLjtEr1QtjDDuKalifd4KdxbXsLa2juqmNprZOHDYhNjSAlKggLhsRw+xRcUweGoXdplu9AyU0NJSGhoYBe/2lS5cyZ84ckpKSgL+fuBkbG3vO523fvp2nn36a5557zqV5Kioq+O53v8uaNWtc8npa9Eqdpri6iVc2FrJiZyklNc0AJEUEMiw2hIkpEQQHOGjv7KKmqZ2SmmYWr83jqU/zSIsJ5u7ZI7h5SjKBfnaL10L119KlSxk/fvypou+rRx99lJ/+9KcuzxMXF0diYiLr169nxowZF/16WvRKAfvK6li8No81ueUYDBNTIlkwMYmpQ6MIDTz7t0lDawc7impYnVvGv7+byx8+PcSjN2dx5ej4QUzvmyoqKrjnnnsoLCwE4Pe//z0zZszgkUceobCwkPz8fAoLC7n//vv50Y9+BMCvf/1rXn31VVJTU4mNjWXq1Kmntt6//e1vExQUxIYNGwD4wx/+wIoVK2hvb+ett95izJgxX3n/+vp6du3axcSJEwF45JFHOHLkCGVlZRw8eJAnnniCjRs3snr1apKTk1mxYgV+fn6kpaXxrW99i7Vr19Le3s6SJUt4+OGHycvL41/+5V+45557ALjxxht59dVXteiVulhFVU387qMDvL+jlGB/O9dnJXBdZgIxoQF9en5ogIOZGbHMGBHDntI6XtpQwPde3MJNk5P55aJMwgP7fqyzJ/jlij3sLa1z6WuOSwrnFwsy+/28H//4xzzwwAPMnDmTwsJCrrvuOvbt2wfA/v37Wbt2LfX19YwePZp7772XnTt38s4777B9+3Y6OjqYMmUKU6dO5dZbb2Xx4sU8/vjjZGdnn3r92NhYtm3bxh//+Ecef/zxM4ZncnJyGD9+/FfuO3z4MGvXrmXv3r187Wtf45133uG3v/0tN910EytXruTGG28EIDU1lQ0bNvDAAw9w5513sn79elpaWsjMzDxV9NnZ2S77bUGLXvmkto4unv1bPk/99RAACyYmsWBiEqEBF/YtISKMT47g0ZuyeG9HCe9tL2F3SS0v3HkJqdHBroyunD755BP27t176uu6ujrq6+sBmD9/PgEBAQQEBBAfH8+xY8dYt24dixYtIigoCIAFCxac8/VvvvlmAKZOncqyZcvOeLysrIy4uK9OFjlv3jz8/PzIysqis7OTuXPnApCVlUVBQcGp5RYuXHjq/oaGBsLCwggLCyMwMJCamhoiIyOJj4+ntLS0n/8qvdOiVz5n69EqHnonl0PHG5iWHs0/TB/W5y348/Gz27htaipjE8L5/V8PsnDxOp674xKmDotyyetb7UK2vAdKV1cXGzZsOFXcpwsI+Pvnabfb6ejowJj+HS118jVOPr+noKCgM45nP/kcm82Gn5/fqUMhbTbbV17j9OVOz3r6ci0tLb2u24XQwyuVz6hraec/3s3llmc2UN3Uxj/PGc0D14xyWcmfbnxyBL9eOJ4Ah53vPr+JrUerXf4evm7OnDksXrz41Nc7duw45/IzZ85kxYoVtLS00NDQwMqVK089FhYWduq3gb4aO3YseXl5/QvdDwcPHjxjaOhCadErn7CloIq5T37Ba5sLuX58Av9z68QB38pOjAzi5wvGERHkxx0vbGZHUc2Avp83a2pqIiUl5dSfJ554gqeeeoqcnBwmTJjAuHHj+NOf/nTO17jkkktYuHAhEydO5OabbyY7O5uIiAgA7rzzTu655x4mTZpEc3NznzKNGTOG2trafv+A6Ku1a9cyf/58l7yW9PfXmYGWnZ1t9MIjylU6Ort46tM8Fn96iLiwAO67MoOM+LBBzVDZ0MqvV+6lua2Td/9pBiPiQgf1/S/Wvn37GDt2rNUxXKKhoYHQ0FCampqYPXs2S5YsYcqUKRf8ek8++SRhYWH84Ac/cGHKbrNnz+b9998nKurMDZLePhMR2WqMyT5jYXSLXnmxoqomvv6/G3nqr4eYkRHLf980YdBLHiAmNIB/nzcWBO5auoWaprZBz6C63X333UyaNIkpU6Zwyy23XFTJA9x7771fGWN3lYqKCh588MFeS/5C6Ba98kprdpfxz2/tossYvj8jnRkZ5z7DcTAcKK/nP1fuZVp6NC99fxp+ds/YzvKmLXpvoVv0yqd1dhkeW7Ofe17ZRkJ4AL+5OcstSh5gdEIYP5g1nC8PV/LY6v1Wx+kXd9sg9GUX8lno4ZXKa1Q3tvF/X9vOurwTXD0mnjsuS3O7rebLR8WRX9HAc+uOMCMjlivHuP8ZtIGBgVRWVupUxW7g5Hz0gYGB/XqeFr3yCrtLavnhy1s5VtfCP84azlVuXKDfvnQYB8rrefDNHay5fzZDwvv3TTvYUlJSKC4upqKiwuooir9fYao/dIxeebwP95Tz49e2ExLg4P5rRpER7/5HtZTUNPMf7+aSPSyKl++6FJvOfKkuko7RK6/18oYC7n1lKynRwfznjeM9ouQBkiOD+O70Yaw/XMlfNhdaHUd5OS165ZGMMfx2zX5+9v4eJg+N4qfzxxIZ7G91rH65akw8WckRPLpqH8XVTVbHUV5Mi155nLaOLn7y1k7++Nlhrh4TzwPXjCLA4XlzwIsI/zgrnS5jeOidXD2yRQ0YLXrlUZrbOrnrpS0s21bCbVNTuGtmukdf2SkuLJDbpw1lXd4Jlm0rsTqO8lJ9KnoRmSsiB0QkT0Qe6uXxABF5w/n4JhFJc97vJyIviUiuiOwTkYddG1/5kpMlv+7QCe6ePZybp6R4xeF+14wdwsj4UP5r1T5qm9qtjqO80HmLXkTswNPAPGAccLuIjOux2F1AtTEmA3gSeMx5/21AgDEmC5gK/PDkDwGl+qOprYPvL93CxvxK7r1ihFddwckmwvdnplPT1Mb/fORZJ1Ipz9CXLfppQJ4xJt8Y0wa8Dizqscwi4CXn7beBq6V7U8sAISLiAIKANsC1l6dRXq+9s4t7X9nGpiOV3HP5CGaNjDv/kzxMWkwIczITeHVjITt1lkvlYn0p+mSg6LSvi5339bqMMaYDqAVi6C79RqAMKAQeN8ZU9XwDEblbRHJEJEdPylCn6+oy/MtbO/n8YAXfn5nulSV/0m1TU4gM9uMXy/fojlnlUn0p+t4GQXv+LzzbMtOATiAJSAd+IiLDz1jQmCXGmGxjTHbPS3Mp3/boqn28t6OUb1ySytVjhlgdZ0AF+zv4enYqO4pqWJlbZnUc5UX6UvTFQOppX6cAPS9keGoZ5zBNBFAFfAtYY4xpN8YcB9YDvZ65pVRPr20u5Ll1R5ibmcCiiUlWxxkUs0fGMSw6mN+s3k9rR6fVcZSX6EvRbwFGiki6iPgD3wSW91hmOXCH8/atwKem+3fPQuAq6RYCTAd0b5M6r435lfzsvd1MTI3gO9OHecXRNX1hswnfunQoxdXNvLzhqNVxlJc4b9E7x9zvAz4E9gFvGmP2iMivRGShc7HngRgRyQMeBE4egvk0EArspvsHxovGmF0uXgflZYqrm7jnla0MCQ/kR1eN9Ojj5C/EhJRIJqZE8NRfD+lFSpRL6KRmyq20d3bx9T9tYH95Pf9103gSI4KsjmSJoqomHlq2i+/NSOdnN/Q8mlmpM+mkZspj/M+HB9heVMPds4f7bMkDpEYHc/moeF76soCjlY1Wx1EeToteuY21+4+z5It8rhk7hOnDY6yOY7nbslOw24TH1uhuLXVxtOiVW6hpauNf397FsOhgvjt9mNVx3EJUsD83TEhiVW45W49WWx1HeTAteuUWHlm+h6qmNu65YgT+Dv1vedINExKJDPLj8Q8PWB1FeTD9jlKW+3BPOe/tKOXGScmkxYRYHcetBPrZWTgpiQ35lXyZd8LqOMpDadErS9U2t/Mf7+aSFhPMjZN946So/rp6zBCiQ/x5/KODOjWCuiBa9MpST358kKrGNu6ePQKHTf879sbfYePGSclsK6zm84M6F5TqP/3OUpbZXVLLnzcUcM3YIaTH6pDNuVw5Oo74sAB+p1v16gJo0StLdHUZfv7+bkIDHdyWnXr+J/g4h93GTZOTyS2p5eO9x6yOozyMFr2yxLvbS9hWWMO3pg0jNMBhdRyPMGtkHIkRgTzx8UG6unSrXvWdFr0adC3tnTz+0QFGxIUwa2Ss1XE8ht0m3Dwlhf3l9azeXW51HOVBtOjVoHvpywLKalu4fdpQbD4yK6WrXDY8hpSoIJ74+ACdulWv+kiLXg2qmqY2nl6bx+ShkWQmRVgdx+PYbMItU1I4XNHIB7t6XhZCqd5p0atB9fTaPBpaO7j9kqFWR/FY09KjSY0KYvGneTpWr/pEi14NmuP1Lfx5w1FmZsSSGh1sdRyPZRNh0aRkDh1v4MM9Olavzk+LXg2aJZ/n097ZxU2TU6yO4vG+NjyGxIhAnvr0kB5Xr85Li14Nior6Vl7ZdJQZGbEkRARaHcfj2WzdW/X7yur5677jVsdRbk6LXg2KZ/+WT1tHFzdNTrY6iteYkRFDfFiAbtWr89KiVwOuqrGNlzcc5bIRsT591ShXc9hsLJyUxK7iWr44pDNbqrPTolcD7s8bCmhu72TRJJ2d0tVmj4wjJsSfp/6qW/Xq7LTo1YBqae/kpS8LmDI0kpQoPdLG1fzsNhZMTGLr0Wo25ldZHUe5KS16NaDe2VZMdVM78yfo1vxAuXJ0PJHBfjz110NWR1FuSoteDZiuLsOzX+QzIi6EsQlhVsfxWv4OGzdkdV+FautR3apXZ9KiVwPm433HKKhsYn5WEqJz2gyoq8fGEx7o4Km/5lkdRbkhLXo1YF5cd4S4sACmpUdbHcXrBfrZuT4rkc8PVrCzqMbqOMrNaNGrAXHwWD0bj1RxzZh47Dbdmh8Mc8YlEBrg4A+f6li9+iotejUg/ryhAD+7cMWYeKuj+Iwgfztzxyfwyb7j7C2tszqOciNa9Mrl6lraeWdbCZeNiCU80M/qOD5lbmYCQX52/viZjtWrv9OiVy63bGsxzW2dXDtuiNVRfE5IgINrxw1h5a4y8isarI6j3IQWvXIpYwx/3nCUjPhQRsSFWh3HJ80bn4Cf3cb/fp5vdRTlJrTolUttPlJF/olGrhmrW/NWiQz25/LRcbyzrZiy2mar4yg3oEWvXOr1LUUE+9uZPlwPqbTSggmJdBnDs18csTqKcgNa9MplapvaWZVbxmUjYglw2K2O49PiwgKZkRHLa5uPUtXYZnUcZTEteuUy7+0oobWji6v0kEq3sHBiEs3tXSxdr1v1vk6LXrmEMYbXNheSHhtCemyI1XEUkBIVzCVpUSz9soD6lnar4ygLadErl9hVXMv+8nquHK1b8+5k0aRk6lo6+MumQqujKAtp0SuXeGdbMf52GzMyYqyOok4zIi6UrOQInv1bPi3tnVbHURbRolcXra2ji+U7S5kyLJJgf4fVcVQPiyYlcaKhjbe2FlsdRVmkT0UvInNF5ICI5InIQ708HiAibzgf3yQiaac9NkFENojIHhHJFZFA18VX7uCLgxXUNLUzKyPO6iiqF+MSw8mID+VPnx2mo7PL6jjKAuctehGxA08D84BxwO0iMq7HYncB1caYDOBJ4DHncx3AK8A9xphM4ApA9wp5mWXbiwkPdDAhNcLqKKoXIsKiSUmU1DSzYlep1XGUBfqyRT8NyDPG5Btj2oDXgUU9llkEvOS8/TZwtXRfaWIOsMsYsxPAGFNpjNGBQi9S29zOJ3uPM314DA6bjgS6qylDo0iNDuKPaw/T1aUXEfc1ffnOTAaKTvu62Hlfr8sYYzqAWiAGGAUYEflQRLaJyL/29gYicreI5IhITkVFRX/XQVloze4y2jq7mDVSh23cmU2EhROTOXS8gU/2HbM6jhpkfSn63q4a0XOT4GzLOICZwLedf98kIlefsaAxS4wx2caY7Lg4LQxPsmxbCYkRgYyI02Pn3d3XhscQHxbA02vzMEa36n1JX4q+GEg97esUoOdA36llnOPyEUCV8/7PjTEnjDFNwCpgysWGVu6huLqJTUeqmJkRq9eE9QB2m3DDhCR2Ftey4XCl1XHUIOpL0W8BRopIuoj4A98ElvdYZjlwh/P2rcCnpnuT4UNggogEO38AXA7sdU10ZbX3d3T/vJ+ZEWtxEtVXl4+KIzLYj8Vr9cIkvuS8Re8cc7+P7tLeB7xpjNkjIr8SkYXOxZ4HYkQkD3gQeMj53GrgCbp/WOwAthljVrp+NdRgM8awbFsxoxPCiA/XI2Y9hb/DxvysRL48XMkOvYi4z+jT2S3GmFV0D7ucft/PT7vdAtx2lue+QvchlsqL7C6p43BFI3fNTLc6iuqnq8cM4f0dpfxxbR5L/iHb6jhqEOjxcOqCLNtejMMmTB+uUx54miB/O3Myh/DR3mPkHa+3Oo4aBFr0qt86OrtYvqOUKUOjCA3QKQ880XXjEvB36OUGfYUWveq3jflVVDa2MUN3wnqs8CA/rhgVx7vbSyivbbE6jhpgWvSq31bmlhHoZ2NSaqTVUdRFmJ/VfbnBF/TCJF5Pi171S0dnFx/uKWdyahT+Dv3v48niwwO5dHgMr246Sm2zTkHlzfQ7VfXL5oIqqhrbuFQv/u0VFkxIorG1k1c3HbU6ihpAWvSqX1bllhHg0GEbb5EeG8KElAheXFegFybxYlr0qs86uwyrd5czKTWSAIfd6jjKRRZMSKKioZV3t5dYHUUNEC161Wc5BVVUNrRxaboO23iTzKRwhseGsOSLfDp1CmOvpEWv+mz17nL87TYmD42yOopyIZHuyc6OnGjk473lVsdRA0CLXvVJV5dhVW4ZE1MjCPTTYRtvc2l6NEPCA3jms8M6hbEX0qJXfbKtsJrj9a1cmq5THngjm02Yn9U9hfGmI1VWx1EupkWv+mRVbjl+dmHyUD3axltdPiqOiCA/nvnssNVRlItp0avzOjlsMyElkmB/ndvGW/k7bFyXmcDnByvYV1ZndRzlQlr06rx2FNdQXteiR9v4gGvHDiHQYWPJFzrZmTfRolfntTq3DIdNmDpMj7bxdqGBDq4aE8/yHaUUVzdZHUe5iBa9OidjDCtzy8hKjtBhGx9xfVYiCDy/Tic78xZa9OqcdhXXUlrTonPb+JCY0AAuGxHD65sLqW5sszqOcgEtenVOq3aXYbcJU4dq0fuSBROSaG7v4uWNOtmZN9CiV2dljGHVrjLGJ4UTGqjDNr4kNTqYyUMjeXH9EZ3szAto0auz2lNaR1F1s54k5aMWTkiiuqmdt3KKrI6iLpIWvTqrVbll2ASmpunRNr5odEIYGfGhPLfuiE525uG06FWvjDGs3FVGZlIE4YF+VsdRFhAR5mclcrSyiU/2HbM6jroIWvSqV/vL6zla1aQnSfm4S9KiiQ8L4Fk9gcqjadGrXp0ctslO06L3ZXabMHd8AjlHq9lRVGN1HHWBtOjVGYwxfLCrjDEJ4UQE6bCNr7tiVDzB/nae+5tu1XsqLXp1hkPHGzhyopHpepKUAoL87Vw1Jp7VueU6LYKH0qJXZ1iVW4bQPT6rFMDczAQQWLq+wOoo6gJo0aszrNxVxuiEMCKD/a2OotxETGgA09OjeW1LIXUt7VbHUf2kRa++Iu94PYeONzB9uJ4kpb5q/oQkGls7eWOznkDlabTo1Veszu2+OLQO26ie0mNDGJcYzgvrj9DR2WV1HNUPWvTqK1bmljF6SBjRITpso850fVYiZbUtrNpdbnUU1Q9a9OqU/IoG9pfX65TE6qwmD40kKTKQZ7/IxxidFsFTaNGrU1Y7t9Km6bCNOgubCPPGJ5JbUsuWgmqr46g+0qJXp6zKLWNkfCgxoQFWR1FubNbIWMICHTyrJ1B5DC16BUBhZRN7SuuYpnPbqPMIcNi5duwQPtl7jCMnGq2Oo/pAi14B3VeSAnTuedUn144bgt0mvKDXlfUIfSp6EZkrIgdEJE9EHurl8QARecP5+CYRSevx+FARaRCRf3ZNbOVqq3LLGBEXQlyYDtuo84sM9mdmRixvby2mpkmvK+vuzlv0ImIHngbmAeOA20VkXI/F7gKqjTEZwJPAYz0efxJYffFx1UAoqmpiV3Et03RrXvXDvKxEmts7+cvmQqujqPPoyxb9NCDPGJNvjGkDXgcW9VhmEfCS8/bbwNUiIgAiciOQD+xxTWTlamucR9vo3POqP4ZGB5OVHMHS9QW0degJVO6sL0WfDJx+znOx875elzHGdAC1QIyIhAD/BvzyXG8gIneLSI6I5FRUVPQ1u3KRVbllpMeGMCQ80OooysNcn5XA8fpWVuWWWR1FnUNfil56ua/nmRJnW+aXwJPGmIZzvYExZokxJtsYkx0XF9eHSMpVSmua2V5Uo0fbqAsyISWS5MggnvubnkDlzvpS9MVA6mlfpwClZ1tGRBxABFAFXAr8VkQKgPuBfxeR+y4ys3Kh1Tpsoy5C9wlUCewurWPzkSqr46iz6EvRbwFGiki6iPgD3wSW91hmOXCH8/atwKem2yxjTJoxJg34PfCoMWaxi7IrF1iVW8awmGASI4KsjqI81KyRcYQFOnheD7V0W+cteueY+33Ah8A+4E1jzB4R+ZWILHQu9jzdY/J5wIPAGYdgKvdTXtvC1qPVOuWBuij+DhvXjB3Cx3uPcbRST6ByR46+LGSMWQWs6nHfz0+73QLcdp7XeOQC8qkBtObkSVI697y6SNeOG8KKnaW8uL6ARxZmWh1H9aBnxvqwD3aVkRodRHKkDtuoixMV7M/XRsTwZk4Rtc16BSp3o0Xvo0prmsk5Ws10PUlKucj1WYk0tXXyup5A5Xa06H3UyeOevzZCi165RlpMCJlJ4Sz9skCvQOVmtOh91IqdpaTHhujRNsql5o3vvgLVar0ClVvRovdBRVVN7Cyu1QuAK5ebPDSSxIhAPYHKzWjR+6APdjmHbfSSgcrFbCLMHZ/AzuJathXqFajchRa9D/pgVykj40OJC9O5bZTrzR4ZR2iAnkDlTrTofcyRE43sKa3TYRs1YAL97Fw1Jp41u8spqmqyOo5Ci97nfLCze5oindtGDaTrMhMQEZZ+WWB1FIUWvc9ZsauUMQlhegFwNaCiQ/yZPjya17cUUt+iJ1BZTYvehxw6Vs/BYw06bKMGxbzxiTS2dvLGlqLzL6wGlBa9D1mxqwyb6LCNGhwj4kIZmximJ1C5AS16H2GM4YOdpYxNDCcy2N/qOMpHzMtMpLi6mY/2HrM6ik/TovcRe8vqyD/RqMM2alBNHRbFkPBAPdTSYlr0PuL9HaXYbaKXDFSDymYT5mYOYevRarbrCVSW0aL3AR2dXby7rYTJqZGEB/pZHUf5mMtHxRPsb9etegtp0fuA9YcrqWhoZdZIvfC6GnxB/t0nUK3eXU5JTbPVcXySFr0PWLatmNAAB5OHRlodRfmo6zITMMbwkp5AZQktei/X0NrBh3vKmT48Gj+7ftzKGrGhAVyaHs1rmwtpbO2wOo7P0e98L7c6t4yW9i4dtlGWuz4rkfqWDt7K0ROoBpsWvZdbtq2EhIhARsaHWh1F+biM+DBGDQnlhfUFdHbpXPWDSYvei5XUNLMhv5JZGbGIiNVxlGLe+EQKq5r4ZJ+eQDWYtOi92HvbSwCYmRFrcRKlul2SFk1cWIAeajnItOi9lDGGZduKGZsQRny4XmBEuQe7TbhuXAKbj1SRW1xrdRyfoUXvpXYV13K4opGZuhNWuZkrx8QR5Gfn+XX5VkfxGVr0XmrZtmL87MJ0vS6scjPB/g6uGB3HB7vKKK9tsTqOT9Ci90JtHV28v7OU7GHRBPs7rI6j1BnmZibQZQwvbSiwOopP0KL3Qp/sO0ZNUzuzRupOWOWe4sMDyR4WzV82FdLUpidQDTQtei/02uZCYkP9mZiiUx4o93V9ViK1ze28vbXY6iheT4veyxRVNbHu0AkuHxWHzabHziv3NWpIKBnxoSz5PF+vQDXAtOi9zJvO08uvGB1vcRKlzk1EWDQpieKaZlbsKrU6jlfTovciHZ1dvLGliImpkcSGBlgdR6nzmjI0itToIP649jBdOi3CgNGi9yKfHajgeH0rV+nWvPIQNhEWTUzm0PEGPtZpEQaMFr0XeWXjUaKC/Zg8THfCKs8xfXgMCeGBLP40D2N0q34gaNF7iYITjXx+sIKrxsTjsOnHqjyH3SbcMDGR3JJa1uWdsDqOV9JG8BKvbDyKzSZcNWaI1VGU6rfZI+OIDvHn6bV5VkfxSlr0XqC5rZM3c4q4JC2K6BB/q+Mo1W9+dhvzsxLZmF/F1qNVVsfxOn0qehGZKyIHRCRPRB7q5fEAEXnD+fgmEUlz3n+tiGwVkVzn31e5Nr4CWL6zhLqWDuaMS7A6ilIX7Kox8YQFOnh67WGro3id8xa9iNiBp4F5wDjgdhEZ12Oxu4BqY0wG8CTwmPP+E8ACY0wWcAfwsquCq27dF1w+Smp0EGMSwqyOo9QFC/SzMzczgU/3H2dvaZ3VcbxKX7bopwF5xph8Y0wb8DqwqMcyi4CXnLffBq4WETHGbDfGnDwTYg8QKHli1fYAAA6fSURBVCJ6gLcLbcivZG9ZHddlJuhVpJTHm5OZQIi/nSc/OWh1FK/Sl6JPBk6/mm+x875elzHGdAC1QEyPZW4BthtjWnu+gYjcLSI5IpJTUVHR1+wKeO6LfMKDHMzK0HnnlecLDXBwfVYiH+89xo6iGqvjeI2+FH1vm4k9D3Y95zIikkn3cM4Pe3sDY8wSY0y2MSY7Lk4Lq6/yjtfz6YEKrh2bgL9D96sr7zBvfCJhgQ5+99EBq6N4jb60QzGQetrXKUDPiSlOLSMiDiACqHJ+nQK8C/yDMUb3srjQ8+sK8LMLc8bpIZXKewT521k4MYm/HTrBpvxKq+N4hb4U/RZgpIiki4g/8E1geY9lltO9sxXgVuBTY4wRkUhgJfCwMWa9q0IrqGxoZdm2YmaNjCM8yM/qOEq51JxxCUQF+/H4Rwf0bFkXOG/RO8fc7wM+BPYBbxpj9ojIr0RkoXOx54EYEckDHgROHoJ5H5AB/ExEdjj/6EQsLvDi+gLaOrq4PivR6ihKuZy/w8aNk5PZUlDNF4f0bNmLJe720zI7O9vk5ORYHcOt1Ta3M+M3n5KZFM7914yyOo5SA6Kjs4sH39pJQnggy++boUeVnYeIbDXGZPf2mO7B80B//rKAhtYObpzc8+AnpbyHw27j5snJ5JbU8tFendnyYmjRe5jG1g6eX3+EKUMjSYsJsTqOUgNq1sg4kiIC+d1HB3S++ougRe9h/rKpkJqmdm6cpFvzyvvZbcItU1M4eKyBZdtLrI7jsbToPUhDawfPfH6Y8cnhjByi0x0o3zB9eAwZcSH8ds1+mto6rI7jkbToPciL645Q1djGN7JTz7+wUl7CJsJ3pqdxvL6VZ784YnUcj6RF7yFqmtr43y/yyR4WRUa8bs0r3zI6IYxL06P50+eHOVbXYnUcj6NF7yH+9Hk+ja0dfF235pWPun3aUNo7u3hs9X6ro3gcLXoPUF7bwtL1R5iREUtqdLDVcZSyxJDwQG6YkMiy7SVsKdCLk/SHFr0HeGzNfjqN4bapKVZHUcpSiyYlExvqz8/e201HZ5fVcTyGFr2b215YzbvbS5iflUh8eKDVcZSyVKCfne9OT2N/eT0vbzxqdRyPoUXvxowx/GrFXiKD/Vg4UY+bVwrgkrQoJqZE8PhHByiv1R2zfaFF78be21HC9qIavnlJKkH+dqvjKOUWRITvzUino9Pw0/dydXbLPtCid1M1TW38+oN9ZMSHMmukXoxFqdMNCQ/k1qkpfLLvOCtzy6yO4/a06N3Uo6v2Udvczg9mpmPTWfuUOsO88YkMjw3h5+/vobqxzeo4bk2L3g1tzK/kzZxi5mclMkwnLlOqV3abcPfs4dQ1t/PT93brEM45aNG7maa2Dh56ZxfxYQHcPEV3wCp1LsNiQrhlagorc8t4Vyc9Oystejfz36v2U1DZxN2zhxPg0B2wSp3PwglJjEkI42fv76aoqsnqOG5Ji96NrD1wnJc3HuX6rEQykyKsjqOUR7DZhP9zxQi6uuD+13fQridSnUGL3k1UNrTyr2/tYmh0sM5OqVQ/xYUF8oNZ6WwtrNa5cHqhRe8GOrsMP3p9OzXNbfyfK0bg79CPRan+umxELHPGDeG5dUdYs1sPuTydNoobeOLjA6zPq+R7M9L1KBulLsJ3pg8jIz6Un7y1k0PH6q2O4za06C320Z5ynl57mCtHx3Pl6Hir4yjl0fzsNu6/eiR+NhvfW7qFyoZWqyO5BS16C+UW1/LjN3YwIi6EOy9LszqOUl4hJjSAn8wZzfG6Vu5+eSutHZ1WR7KcFr1Fiqqa+P7SLYT42/nJnNE6Lq+UC2XEh3LvFSPYerSaB97YQWeXb59M5bA6gC+qbGjlzhc309TWwSMLM4kK9rc6klJeZ/rwGCob2nhl01FC/Hfx2C0TsNl8czoRLfpBVtXYxref20RRVTP/Nm8MKVF6xSilBsr8CYk0t3fw1tZiQgIc/GLBOMQH547Soh9E1Y1tfOe5TRyuaOCf54xmXGK41ZGU8nq3TEmhub2LpV8W0NrRyX/emIXdx7bstegHSXF1E3e8sJnCqiZ+cu1oJqREWh1JKZ8gInzn0qEEOGy8trmI+pYOnvj6JJ/aL6ZFPwj2lNZy54tbuicsmzdWt+SVGmQiwtezUwn2t/PqpkIq6lt55jtTiQ7xjf1jvvMjzSLv7yjh1mc2YIzhFzdkaskrZaEbJiRx35UZbCusZuHidRwo942TqrToB0hrRye/eH83P359B8NigvnVovGkRuuOV6WsNiMjlp/fkElDaweLFq/jtc2FXj+XvRb9ANhZVMP8p9bx0obumSj/Y/5YPYRSKTeSER/Kf9+UxcghYTy8LJd7X9lKRb33nkUr7vaTLDs72+Tk5Fgd44LUtbTz1CeHeGH9EaKC/fnBrOFMStWdrkq5qy5jWLmrjDdzigj2t/Pw9WP5RnaqRx5vLyJbjTHZvT6mRX/x2ju7eDOniN99dJDqxjauGB3Pd6YPJdhf93Ur5QlKapp5YV0+e8vqGZ8czsPzxjIjI9bqWP2iRT9AWto7eXtrMc98dpiSmmbGJITx3enDGB4XanU0pVQ/GWNYl3eCN3OKONHQxmUjYrjn8hHMGhnrESdZadG72JETjby2uZA3thRR29zOyPhQbpqczKTUSI/4D6GUOru2ji4+2XeMD3aVUt3UztjEML596TAWTEwiIsjP6nhnpUV/kYwxHDnRyCf7jrFiZxm5JbXYBLLTopkzbgjjEsO14JXyMu2dXazPO8Ga3eUcrWoiwGHjmrFDmDs+gSvHxBMa4F5Ds1r0/dTVZSiubmZrYRVbCqpZd+gEhc6LDg+PDeFrI2K4bESsz5xsoZQvM8aQf6KRzw5UsKWgitrmdhw2YcqwKGaMiGXKsEgykyIs74OLLnoRmQv8P8AOPGeM+U2PxwOAPwNTgUrgG8aYAudjDwN3AZ3Aj4wxH57rvQar6I0x1Da3U17XQlltC/kVjRwsr+fAsXoOHqunqa17DutgfztjEsKYmBrJpJRI4sMDBzybUso9dXUZDh6rZ1thNbtL6yg40cjJBk2KDCQzKYLMpHCGxQSTFBFEclQQCeGBOOwDfyT7uYr+vL97iIgdeBq4FigGtojIcmPM3tMWuwuoNsZkiMg3gceAb4jIOOCbQCaQBHwiIqOMMS6/EkBFfStrdpfR1mno6OyivbOLtk5Da0cn9S0d1Ld0UNfcTn1LO5WNbRyra6Gl/atXiw8PdJAaHcyskXGkRgWRER9KalSwRx5qpZRyPZtNGJMYzhjnGe4NrR0UnGjkyIlGCiob2VNayyd7j3H65rNdhLiwAKJC/IgM8iciyI/IYD9CAxz4O2wEOOz4O2z4O2xkxIdy+ag4l+fuyyDTNCDPGJMPICKvA4uA04t+EfCI8/bbwGLpHrReBLxujGkFjohInvP1Nrgm/t+V1Tbzs/f3nHG/n10ICXAQ4u8g2N9OSICDodHBTE6NJCY0gJgQf2JDA0iICNSTmpRS/RIa4CAhPJDpw2NO3dfa0UlFfSvH61upqG/lWF0LlY1tNLR0UNvcTklNMw2tHTS1ddDeab5yUZQFE5MsK/pkoOi0r4uBS8+2jDGmQ0RqgRjn/Rt7PDe55xuIyN3A3c4vG0TkwHkyxQIn+pDdW+j6ejdfW1/wvXXu0/ouBhZ/64LfY9jZHuhL0fc2btFzYP9sy/TluRhjlgBL+pCl+81Ecs42FuWNdH29m6+tL/jeOlu9vn3ZQ1AMpJ72dQpQerZlRMQBRABVfXyuUkqpAdSXot8CjBSRdBHxp3vn6vIeyywH7nDevhX41HQfzrMc+KaIBIhIOjAS2Oya6EoppfrivEM3zjH3+4AP6T688gVjzB4R+RWQY4xZDjwPvOzc2VpF9w8DnMu9SfeO2w7gn1x0xE2fh3m8hK6vd/O19QXfW2dL19ftTphSSinlWjofvVJKeTkteqWU8nIeVfQiUiAiuSKyQ0Tca+YzFxGRF0TkuIjsPu2+aBH5WEQOOf+OsjKjK51lfR8RkRLn57xDRK63MqMriUiqiKwVkX0iskdEfuy83ys/43Osr1d+xiISKCKbRWSnc31/6bw/XUQ2OT/fN5wHtgxeLk8aoxeRAiDbGOO1J1qIyGygAfizMWa8877fAlXGmN+IyENAlDHm36zM6SpnWd9HgAZjzONWZhsIIpIIJBpjtolIGLAVuBG4Ey/8jM+xvl/HCz9j54wAIcaYBhHxA9YBPwYeBJYZY14XkT8BO40xzwxWLo/aovcFxpgv6D5y6XSLgJect1+i+xvFK5xlfb2WMabMGLPNebse2Ef32eJe+RmfY329kunW4PzSz/nHAFfRPT0MWPD5elrRG+AjEdnqnDbBVwwxxpRB9zcOEG9xnsFwn4jscg7teMUwRk8ikgZMBjbhA59xj/UFL/2MRcQuIjuA48DHwGGgxhjT4Vyk16lgBpKnFf0MY8wUYB7wT85f+5X3eQYYAUwCyoDfWRvH9UQkFHgHuN8YU2d1noHWy/p67WdsjOk0xkyieyaAacDY3hYbzEweVfTGmFLn38eBd+n+R/QFx5xjnSfHPI9bnGdAGWOOOb9ZuoBn8bLP2Tl2+w7wqjFmmfNur/2Me1tfb/+MAYwxNcBnwHQg0jk9DFgwFYzHFL2IhDh35iAiIcAcYPe5n+U1Tp9i4g7gfQuzDLiThed0E170OTt31j0P7DPGPHHaQ175GZ9tfb31MxaROBGJdN4OAq6he7/EWrqnhwELPl+POepGRIbTvRUP3VM3/MUY818WRhoQIvIacAXd05oeA34BvAe8CQwFCoHbjDFesQPzLOt7Bd2/0hugAPjhyfFrTyciM4G/AbnAySvf/Dvd49Ze9xmfY31vxws/YxGZQPfOVjvdG9JvGmN+5eyv14FoYDvwHed1OgYnl6cUvVJKqQvjMUM3SimlLowWvVJKeTkteqWU8nJa9Eop5eW06JVSystp0SullJfToldKKS/3/wH5iwETyGhEEgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sb\n",
    "sb.kdeplot(data['Length (mm)'], shade = True);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### How to calculate information gain (entropy)? - how do we quantify the quality of data split?\n",
    "\n",
    "#### Formule of information gain\n",
    "Entropy (_parent_) - m/(m+n)*entropy(child1)- n/(m+n)entropy(child2...)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Calculate parent entropy\n",
    "- two types of species (Mobug vs. Lobug)\n",
    "- Mobug=10 and Lobug =14\n",
    "- entropy = -(p1log2p1+p2log2p2) while p1 = m/m+n\n",
    "let's m = Mobug and n = Lobug\n",
    "p1 = 10/(10+14)\n",
    "p2 = 14/(10+14)"
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
      "0.9798687566511528\n"
     ]
    }
   ],
   "source": [
    "# entropy parent\n",
    "p1 = 10/(10+14)\n",
    "p2 = 14/(10+14)\n",
    "S_parent = -(p1*np.log2(p1)+p2*np.log2(p2))\n",
    "print(S_parent)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mobug    4\n",
      "Lobug    2\n",
      "Name: Species, dtype: int64\n",
      "Lobug    6\n",
      "Mobug    2\n",
      "Name: Species, dtype: int64\n",
      "Lobug    6\n",
      "Mobug    4\n",
      "Name: Species, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# determine how many lobugs and mobugs if we split them based on their colors\n",
    "print(data[data['Color']=='Brown'].Species.value_counts())\n",
    "print(data[data['Color'] == 'Green'].Species.value_counts())\n",
    "print(data[data['Color'] == 'Blue'].Species.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [],
   "source": [
    "# entrop of brown child\n",
    "\n",
    "def two_ent(first, tot):\n",
    "    return -(first/tot*np.log2(first/tot) + \n",
    "             (tot-first)/tot * np.log2((tot - first)/tot))\n",
    "\n",
    "S_brown = two_ent(4,6)\n",
    "S_green = two_ent(6,8)\n",
    "S_blue = two_ent(6, 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9182958340544896 0.8112781244591328 0.9709505944546686\n"
     ]
    }
   ],
   "source": [
    "print(S_brown, S_green, S_blue)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.06157292259666325\n",
      "0.16859063219201997\n",
      "0.008918162196484225\n"
     ]
    }
   ],
   "source": [
    "# gain entropy based on color split\n",
    "print(S_parent - S_brown)\n",
    "print(S_parent - S_green)\n",
    "print(S_parent - S_blue)"
   ]
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
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
