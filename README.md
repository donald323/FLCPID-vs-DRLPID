# FLCPID-vs-DRLPID

Greeting to fellow readers who are interested in Fuzzy PID and/or Deep Reinforcement Learning-based PID, and/or autonomous ship simulation. I am Donald Cheng, a master student from Xi'an Jiaotong-Liverpool University who study Pattern Recognition and Intelligent Systems, and my dissertation project is a comparative analysis of Fuzzy Logic and Deep Reinforcement Learning’s applications in PID Tuning for autonomous ships. I will be sharing my codes in how I apply Fuzzy Logic/Deep Reinforcement Learning in building an adaptive PID to improve an autonomous ship's performance in error reduction. The codes are in Python, and I will be continue to update this repositories. In addition, I will also upload my reference list used for this project, in case you want to know equations behind the codes. Last but not least, feel free to use my code and post any comments on this page, or contact me via donald_cheng1@hotmail.com if you have any questions.

#Measures of the Results

Errors Performance:

Unacceptable: 2.0 <= average AAE

Poor: 1.5 <= average AAE < 2.0

Average: 1.0 <= average AAE < 1.5

Good: 0.5 <= average AAE < 1.0

Excellent: average AAE <= 0.5

Stability:

Unstable: 20% <= %RD

Slightly stable: 15% <= %RD < 20%

Moderately Stable: 10% <= %RD < 15%

Stable: 5% <= %RD < 10%

Very Stable: %RD  <= 5%


# Notes

Please make sure to copy all .py file into the same folder to ensure the simulator to work.

# Libraries
- Ship Simulator
  - python 3.7.2
  - numpy 1.21.5
  - statsmodels 0.13.2
  - matplotlib 3.5.1
- Fuzzy Logic
  - scikit-fuzzy 0.4.2
- Deep Reinforcement Learning (RL_DQN, RL_A2C, RL_DDPG)
  - pytorch 1.4.0
  - cudatoolikit 10.1.243 (please install CUDA toolkit that fits your NVIDIA graphic card for Deep Learning (I don't have solutions for other graphic cards), or you can also just use CPU to run the Deep Reinforcement Learning)

# Updates

11.17.2022

- Added libraries and a demo file ('ship_simulation_demo.py') for the ship simulator.
- Update README.md

12.5.2022

- Upload all codes of ship simulator with AIPID and plots of all results, with collected data for full test and target test.
