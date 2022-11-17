#File for Fuzzy Logic Control system for adjusting PID gains with Scikit-Fuzzy
#Membership Function and Rule Bases are based on the paper 'An Adaptive, Robust control of DC motor using Fuzzy-PID controller', by Rishabh Abhinav and Satya Sheel

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

#Fuzzy Related Parameters
universe = np.linspace(-1,1,7)
flc_e = ctrl.Antecedent(np.arange(-1,4/3,1/3),'Error')
flc_ed = ctrl.Antecedent(np.arange(-1,4/3,1/3),'Error Derivative')
kp_out = ctrl.Consequent(np.arange(-1,4/3,1/3),'Kp Output', defuzzify_method='centroid')
ki_out = ctrl.Consequent(np.arange(-1,4/3,1/3),'Ki Output', defuzzify_method='centroid')
kd_out = ctrl.Consequent(np.arange(-1,4/3,1/3),'Kd Output', defuzzify_method='centroid')

names = ['NL','NM','NS','ZE','PS','PM','PL']

flc_e.automf(names=names)
flc_ed.automf(names=names)
kp_out.automf(names=names)
ki_out.automf(names=names)
kd_out.automf(names=names)

kp_rule1 = ctrl.Rule(antecedent=(flc_e['NL'] | flc_e['PL']),
   consequent=kp_out['PL'], label='Kp PL Rule')
kp_rule2 = ctrl.Rule(antecedent=((flc_e['NM'] | flc_e['PM']) & 
   (flc_ed['NM'] | flc_ed['NS'] | flc_ed['ZE'] | flc_ed['PS'] | flc_ed['PM'])),
   consequent=kp_out['PM'], label='Kp PM Rule')
kp_rule3 = ctrl.Rule(antecedent=(((flc_e['NM'] | flc_e['PM']) & (flc_ed['NL'] | flc_ed['PL'])) | 
   ((flc_e['NS'] | flc_e['PS']) & (flc_ed['NS'] | flc_ed['ZE'] | flc_ed['PS']))),
   consequent=kp_out['PS'], label='Kp PS Rule')
kp_rule4 = ctrl.Rule(antecedent=(((flc_e['NS'] | flc_e['PS']) & (flc_ed['NL'] | flc_ed['NM'] | flc_ed['PM'] | flc_ed['PL'])) | 
   (flc_e['ZE'] & (flc_ed['NS'] | flc_ed['ZE'] | flc_ed['PS']))),
   consequent=kp_out['ZE'], label='Kp ZE Rule')
kp_rule5 = ctrl.Rule(antecedent=(flc_e['ZE'] & (flc_ed['NL'] | flc_ed['PL'])),
   consequent=kp_out['NL'], label='Kp NL Rule')
kp_rule6 = ctrl.Rule(antecedent=(flc_e['ZE'] & (flc_ed['NM'] | flc_ed['PM'])),
   consequent=kp_out['NS'], label='Kp NS Rule')

ki_rule1 = ctrl.Rule(antecedent=((flc_e['NL'] | flc_e['PL']) |
	flc_ed['ZE'] |
	((flc_e['NM'] | flc_e['PM']) & (flc_ed['NM'] | flc_ed['NS'] | flc_ed['ZE'] | flc_ed['PS'] | flc_ed['PM']))),
	consequent=ki_out['NL'], label='Ki NL Rule')
ki_rule2 = ctrl.Rule(antecedent=(((flc_e['NM'] | flc_e['PM']) & (flc_ed['NL'] | flc_ed['PL'])) | 
	((flc_e['NS'] | flc_e['PS']) & (flc_ed['NS'] | flc_ed['PS']))), 
	consequent=ki_out['NM'], label='Ki NM Rule')
ki_rule3 = ctrl.Rule(antecedent=((flc_e['NS'] | flc_e['PS']) & (flc_ed['NM'] | flc_ed['PM'])),
	consequent=ki_out['ZE'], label='Ki ZE Rule')
ki_rule4 = ctrl.Rule(antecedent=(flc_e['ZE'] & (flc_ed['NS'] | flc_ed['PS'])),
	consequent=ki_out['PS'], label='Ki PS Rule')
ki_rule5 = ctrl.Rule(antecedent=(((flc_e['NS'] | flc_e['PS']) & (flc_ed['NL'] | flc_ed['PL'])) |
	(flc_e['ZE'] & (flc_ed['NM'] | flc_ed['PM']))),
	consequent=ki_out['PM'], label='Ki PM Rule')
ki_rule6 = ctrl.Rule(antecedent=(flc_e['ZE'] & (flc_ed['NL'] | flc_ed['PL'])),
	consequent=ki_out['PL'], label='Ki PL Rule')

kd_rule1 = ctrl.Rule(antecedent=(flc_e['NL'] | flc_e['PL']),
	consequent=kd_out['NL'], label='Kd NL Rule')
kd_rule2 = ctrl.Rule(antecedent=((flc_e['NM'] & (flc_ed['NS'] | flc_ed['ZE'] | flc_ed['PS'])) | 
	(flc_e['PM'] & (flc_ed['NS'] | flc_ed['PS']))),
	consequent=kd_out['NM'], label='Kd NM Rule')
kd_rule3 = ctrl.Rule(antecedent=((flc_e['NM'] | flc_e['PM']) & (flc_ed['NM'] | flc_ed['PM'])),
	consequent=kd_out['NS'], label='Kd NS Rule')
kd_rule4 = ctrl.Rule(antecedent=(((flc_e['NL'] | flc_e['PL']) & (flc_ed['NL'] | flc_ed['PL'])) | 
	((flc_e['NS'] | flc_e['PS']) & (flc_ed['NM'] | flc_ed['NS'] | flc_ed['ZE'] | flc_ed['PS'] | flc_ed['PM']))),
	consequent=kd_out['ZE'], label='Kd ZE Rule')
kd_rule5 = ctrl.Rule(antecedent=(((flc_e['NS'] | flc_e['PS']) & (flc_ed['NL'] | flc_ed['PL'])) | 
	(flc_e['ZE'] & flc_ed['ZE'])),
	consequent=kd_out['PS'], label='Kd PS Rule')
kd_rule6 = ctrl.Rule(antecedent=(flc_e['ZE'] & (flc_ed['NS'] | flc_ed['PS'])),
	consequent=kd_out['PM'], label='Kd PM Rule')
kd_rule7 = ctrl.Rule(antecedent=(flc_e['ZE'] & (flc_ed['NL'] | flc_ed['NM'] | flc_ed['PM'] | flc_ed['PL'])),
	consequent=kd_out['PL'], label='Kd PL Rule')

kp_system = ctrl.ControlSystem(rules=[kp_rule1,kp_rule2,kp_rule3,kp_rule4,kp_rule5,kp_rule6])
kp_sim = ctrl.ControlSystemSimulation(kp_system)
ki_system = ctrl.ControlSystem(rules=[ki_rule1,ki_rule2,ki_rule3,ki_rule4,ki_rule5,ki_rule6])
ki_sim = ctrl.ControlSystemSimulation(ki_system)
kd_system = ctrl.ControlSystem(rules=[kd_rule1,kd_rule2,kd_rule3,kd_rule4,kd_rule5,kd_rule6,kd_rule7])
kd_sim = ctrl.ControlSystemSimulation(kd_system)