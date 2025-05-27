### Recurrent Neural Networks (RNNs)  

#### What is an RNN?  
A **Recurrent Neural Network (RNN)** is a type of neural network that is designed for **sequential data**. Unlike traditional neural networks (like feedforward networks), RNNs have a built-in "memory," allowing them to **remember previous inputs** while processing new ones.  

#### Why Do We Need RNNs?  
Many real-world problems involve **sequences** where the order of data matters. Examples include:  
- **Text data** (predicting the next word in a sentence)  
- **Speech recognition** (processing spoken words over time)  
- **Time series forecasting** (predicting stock prices based on past trends)  
- **Video processing** (analyzing frames over time)  

A traditional neural network treats inputs independently, but RNNs understand the **context** by remembering past information.  

#### How Does an RNN Work?  
- Instead of processing an entire input at once, RNNs process **one step at a time** and pass information forward.  
- Each time the network sees a new input, it **updates its hidden state** based on both the new input and the previous hidden state.  
- This allows the network to capture dependencies in sequential data.  

#### RNN Structure  
An RNN has:  
1. **Input Layer** – Receives one part of the sequence at a time.  
2. **Hidden Layer(s)** – Maintains memory of past inputs using a loop structure.  
3. **Output Layer** – Produces predictions based on hidden states.  

#### Mathematical Representation  
Each step in an RNN follows this formula:  
$$
h_t = f(W_h h_{t-1} + W_x x_t + b)
$$
where:  
- $ h_t $ = hidden state at time $ t $  
- $ W_h $, $ W_x $ = weight matrices  
- $ x_t $ = input at time $ t $  
- $ b $ = bias  
- $ f $ = activation function (usually **tanh** or **ReLU**)  

The output is often computed as:  
$$
y_t = W_y h_t + b_y
$$  

#### Problems with RNNs  
**Vanishing Gradient Problem**: When training deep RNNs, gradients (used for learning) can become **very small**, making it hard for the network to learn long-term dependencies.  

Solution: **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** networks were developed to solve this problem!  

#### RNN Example in Python (Using TensorFlow)  
Here’s how to create a simple RNN using **Keras (TensorFlow)**:  
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Define an RNN model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(10, 1)),  # 50 neurons, input sequence of 10 time steps
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Model summary
model.summary()
```

#### Summary  
- RNNs are designed for **sequential data**.  
- They **remember past information** and use it to predict future steps.  
- They suffer from the **vanishing gradient problem**, which LSTMs and GRUs help fix.  
- They are widely used in **NLP, speech recognition, and time series forecasting**.  

--------
### **How LSTM Works**  

**Long Short-Term Memory (LSTM)** is a special type of **Recurrent Neural Network (RNN)** designed to handle **long-range dependencies** and prevent issues like the **vanishing gradient problem**. LSTMs achieve this using **gates** that control the flow of information.  

---

### LSTM Architecture
         ┌──────────────┐    
         │  Forget Gate │   Decides what to discard  
         └──────────────┘    
                ↓  
         ┌──────────────┐    
         │  Input Gate  │   Decides what to store  
         └──────────────┘    
                ↓  
       ┌─────────────────┐  
       │  Cell State (Ct)│   Long-term memory  
       └─────────────────┘  
                ↓  
         ┌──────────────┐  
         │  Output Gate │   Decides what to pass forward  
         └──────────────┘  
                ↓  
         **New Hidden State (ht)**   


## ** LSTM Cell Structure**
Each LSTM cell has three main components:  
1. **Cell state ($ C_t $)** → Memory of the network, which flows through time.  
2. **Hidden state ($ h_t $)** → Short-term memory, output at each step.  
3. **Gates (Forget, Input, Output)** → Control what information is kept or discarded.

---

## ** LSTM Step-by-Step Computation**
### **Step 1: Forget Gate**
**Decides what past information to discard**  

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

- $ f_t $ → Forget gate output (0 = forget, 1 = keep)  
- $ W_f $, $ b_f $ → Learnable weights & bias  
- $ h_{t-1} $ → Previous hidden state  
- $ x_t $ → Current input  
- $ \sigma $ → Sigmoid activation (outputs values between 0 and 1)  

If $ f_t $ is close to **1**, the memory is **kept**. If close to **0**, it is **forgotten**.

---

### **Step 2: Input Gate**
**Decides what new information to add to the memory**  

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$

- $ i_t $ → Input gate output (how much new info to store)  
- $ \tilde{C}_t $ → Candidate memory (potential update to cell state)  
- $ C_t $ → Updated cell state  

The **forget gate** controls old memory, and the **input gate** adds new information.

---

### **Step 3: Output Gate**
**Decides what part of the memory to output as the hidden state**  

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \cdot \tanh(C_t)
$$

- $ o_t $ → Output gate activation  
- $ h_t $ → Updated hidden state (what is passed to the next time step)  

The final hidden state $ h_t $ is used for predictions.

---

## ** LSTM Step-by-Step Flow**
- **Step 1: Forget Gate** → Forget irrelevant past info  
- **Step 2: Input Gate** → Add relevant new info  
- **Step 3: Update Cell State** → Store long-term memory  
- **Step 4: Output Gate** → Output relevant info  

---

## ** LSTM vs. Simple RNN**
| Feature | Simple RNN | LSTM |
|---------|-----------|------|
| Handles long sequences? |  No |  Yes |
| Vanishing gradient issue? |  Yes |  No |
| Memory mechanism? |  No |  Yes (Cell State) |
| Used in NLP, Speech, Time-Series? | Limited |  Preferred |

---

### **Conclusion**
LSTMs are **powerful** because they **store important past information** while **discarding irrelevant details**.  
They **prevent vanishing gradients**, making them great for **long text, speech, and time-series data**.  

