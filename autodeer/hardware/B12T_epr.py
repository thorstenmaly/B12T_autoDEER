from autodeer.classes import  Interface, Parameter
from autodeer.pulses import Pulse, RectPulse, ChirpPulse, HSPulse, Delay, Detection
from autodeer.sequences import *
from autodeer.FieldSweep import create_Nmodel
from autodeer.dataset import get_all_fixed_param, create_dataset_from_sequence
import yaml

from .dummy import  _simulate_CP, _simulate_T2, _gen_ESEEM, add_phaseshift, _simulate_field_sweep, _similate_respro, _simulate_reptimescan, _simulate_2D_relax, _simulate_deer, add_noise
import xarray as xr
import datetime
import numpy as np
import deerlab as dl
import time
import socket
import logging

import dnplab as dnp
import time
import os

hw_log = logging.getLogger('interface.B12TEpr')
verbosity = 3

## If True, the program will load fake data
SKIP_OPT_PULSE = False
SKIP_FSWEEP = False
SKIP_REPTIME = False
SKIP_RESPRO = False
SKIP_CP = False
SKIP_T2 = False
SKIP_2D = False
SKIP_QUICK_DEER = False
SKIP_LONG_DEER = False
DUMMY_FSWEEP = 'FieldSweepSequence_20241213_0952_ascn000' #'FieldSweepSequence_dummy'
DUMMY_REPTIME = 'ReptimeScan_20241213_0942_ascn000' #'ReptimeScan_dummy'
DUMMY_RESPRO = 'ResonatorProfileSequence_20241213_0944_ascn000' # 'ResonatorProfileSequence_dummy'
DUMMY_CP = 'CP_dummy'
DUMMY_T2 = 'T2_dummy'
DUMMY_2D = 'RefocusedEcho2D_dummy'
DUMMY_QUICK_DEER = 'Quick_4pDEER_dummy'
DUMMY_LONG_DEER = ''

class B12TInterface(Interface):

    def __init__(self,config_file) -> None:
        with open(config_file, mode='r') as file:
            config = yaml.safe_load(file)
            self.config = config    

        # Dummy = config['Spectrometer']['Dummy']
        self.Bridge = config['Spectrometer']['Bridge']
        resonator_list = list(config['Resonators'].keys())
        key1 = resonator_list[0]
        self.fc = self.config['Resonators'][key1]['Center Freq'] * 1e9 # Hz
        self.Q = self.config['Resonators'][key1]['Q']
        self.resonator_bandwidth = self.config['Resonators'][key1]['bandwidth'] * 1e6 # Hz

        # AWG:
        self.awg_state = config['Spectrometer']['AWG']['state']
        self.awg_freq = config['Spectrometer']['AWG']['frequency'] * 1e6 # Hz

        # receiver amplifier
        self.rcv_state = config['Spectrometer']['RCV']['state']

        # video amplifier
        self.vva_state = config['Spectrometer']['VideoAMP']['state']
        self.vva_gain = config['Spectrometer']['VideoAMP']['gain']

        # high power amplifier
        self.hamp_state = config['Spectrometer']['HP_AMP']['state']

        # Sample
        self.sample_freq = config['Spectrometer']['Bridge']['Sample Freq'] * 1e9 # Hz
        self.sample_field = config['Spectrometer']['Bridge']['Sample Field']
        self.field_to_freq = (self.sample_freq + self.awg_freq) / self.sample_field

        # initial field
        ## self.fc is the set frequency to Bridge, and self.field is correlated field at that frequency
        self.field = config['Spectrometer']['Field']['Field'] if config['Spectrometer']['Field']['Field'] else (self.fc + self.awg_freq)/self.field_to_freq
        self.field_unit = config['Spectrometer']['Field']['Field_unit'] if config['Spectrometer']['Field']['Field_unit'] else 'G'

        self.saving_path = config['Specman']['Saving_path']

        # relaxataion time
        self.guess_tau1 = config['initial_params']['guess_tau1'] * 1e-9
        self.guess_tau2 = config['initial_params']['guess_tau2'] * 1e-9
        self.tau1_fix = config['initial_params']['fix_tau1']
        self.tau2_fix = config['initial_params']['fix_tau2']
        self.tau1 = self.guess_tau1 if self.tau1_fix else None # s
        self.tau2 = self.guess_tau1 if self.tau2_fix else None # s

        # optimize pulse:
        self.t90 = config['initial_params']['t90'] * 1e-9
        self.t180 = config['initial_params']['t180'] * 1e-9
        self.t_nut = config['initial_params']['t_nut'] * 1e-9

        self.optimze_list = ['Observe', 'Refocus', 'Pump']
        self.amp90 = 0.1
        self.amp180 = 1.0
        self.amp_nut = 1.0

        self.pulse90 = None
        self.pulse180 = None
        # DEER X Axis
        self.t_start = config['initial_params']['t_start'] * 1e-9
        self.t_step = config['initial_params']['t_step'] * 1e-9
        self.t_size = config['initial_params']['t_size'] 
        self.t_stop = self.t_start + (self.t_size - 1) * self.t_step

        # For SNR
        self.current_scan_file = None

        # For 


        # For Experiment Parameters
        self.LO = self.fc # Hz
        self.scans = 1
        self.shots = 1
        self.reptime = 1e-3 # s
        self.ncyc = 1
        
        super().__init__(log=hw_log)
         
    def connect(self):
        self._connect_epr()
        self.send_message(".server.opmode = 'Operate'")
        self.log_message('SpecMan4EPR: Setting...', verbosity)
        self.log_message('SpecMan4EPR: Frequency: %0.04f GHz' %(self.fc*1e-9), verbosity)
        self.send_message(".spec.BRIDGE.Frequency = %0.04f" %(self.fc))
        self.log_message('SpecMan4EPR: RecvAmp: %i' %(self.rcv_state), verbosity)
        self.send_message(".spec.BRIDGE.RecvAmp = %i" %(self.rcv_state))
        self.log_message('SpecMan4EPR: VideoAmp %i' %(self.vva_state), verbosity)
        self.send_message(".spec.BRIDGE.VideoAmp = %i" %(self.vva_state))
        self.log_message('SpecMan4EPR: VideoGain: %i' %(self.vva_gain), verbosity)
        self.send_message(".spec.BRIDGE.VideoGain = %i" %(self.vva_gain))
        self.send_message(".spec.BRIDGE.IO2 = %i" %(self.hamp_state))
        self.send_message(".spec.FLD.Field = %0.05f" %(self.field))
        self.log_message('SpecMan4EPR: Field: %0.05f %s' %(self.field, self.field_unit), verbosity)
        wait(10)
        self.log_message('SpecMan4EPR: Done', verbosity)
        self.log_message('SpecMan4EPR: Connected to SpecMan', verbosity)
        return super().connect()
        
    def launch(self, sequence, savename: str, **kwargs):
        self.log_message(f"Launching {sequence.name} sequence", verbosity)
        self.state = True
        self.cur_exp = sequence
        self.start_time = time.time()
        self.fguid = self.cur_exp.name + "_" +  datetime.datetime.now().strftime('%Y%m%d_%H%M')
        # self.fguid = self.cur_exp.name
        if isinstance(self.cur_exp, FieldSweepSequence):
            if not SKIP_FSWEEP:
                self._run_fsweep(self.cur_exp)
            else:
                wait(30)
                self.fguid = DUMMY_FSWEEP
            pass
        elif isinstance(self.cur_exp, ReptimeScan):
            if not SKIP_REPTIME:
                self._run_reptimescan(self.cur_exp)
            else:
                wait(30)
                self.fguid = DUMMY_REPTIME
            pass
        elif isinstance(self.cur_exp, ResonatorProfileSequence):
            if not SKIP_RESPRO:
                self._run_respro(self.cur_exp)
            else:
                wait(30)
                self.fguid = DUMMY_RESPRO
            pass
        elif isinstance(self.cur_exp, DEERSequence):
            if self.cur_exp.t.is_static():
                self.log_message("It is Carpell", verbosity)
                if not SKIP_CP:
                    self._run_CP_relax(self.cur_exp)
                else:
                    wait(30)
                    self.fguid = DUMMY_CP
            else:
                if self.cur_exp.name == '5pDEER':
                    deer_method = self._run_5pdeer
                elif self.cur_exp.name == '4pDEER':
                    deer_method = self._run_4pdeer
                else:
                    raise NameError('%s is not supported' %self.cur_exp.name)
                
                if self.cur_exp.averages.value < 1000: # quick DEER
                    self.log_message("It is quick DEER", verbosity)
                    if not SKIP_QUICK_DEER:
                        self.fguid = 'Quick_' + self.fguid
                        deer_method(self.cur_exp)
                    else:
                        wait(10)
                        self.fguid = DUMMY_QUICK_DEER
                else: # long DEER
                    self.log_message("It is long DEER", verbosity)
                    if not SKIP_LONG_DEER:
                        deer_method(self.cur_exp)
                    else:
                        wait(10)
                        self.fguid = DUMMY_LONG_DEER

        elif isinstance(self.cur_exp, T2RelaxationSequence):
            if not SKIP_T2:
                self._run_T2_relax(self.cur_exp)
            else:
                wait(30)
                self.fguid = DUMMY_T2

        elif isinstance(self.cur_exp, RefocusedEcho2DSequence):
            if not SKIP_2D:
                self._run_2D_relax(self.cur_exp)
            else:
                wait(30)
                self.fguid = DUMMY_2D

        return super().launch(sequence, savename)
    
    def acquire_dataset(self, **kwargs):
        if self.fguid:
            if not self.isrunning():
                if self.fguid + '.exp' in os.listdir(self.saving_path):
                    self.log_message("SpecMan4EPR: Analyzing Final %s dataset..." %self.cur_exp.name, verbosity)
                    dset = self._create_dataset_from_b12t(self.saving_path + self.fguid + '.exp')
                    self.log_message("SpecMan4EPR: Analyze done %s" %self.cur_exp.name, verbosity)
                    self.current_scan_file = None

            else:
                while True:
                    scan_files = [x for x in os.listdir(self.saving_path) if (self.fguid in x and x != self.fguid + '.exp' and '~' not in x and '.exp' in x)]
                    if scan_files and scan_files[-1] != self.current_scan_file:
                        self.current_scan_file = scan_files[-1]
                        time.sleep(1)
                        self.log_message("SpecMan4EPR: Analyzing %s scan #%i dataset" %(self.cur_exp.name, len(scan_files)), verbosity)
                        dset = self._create_dataset_from_b12t(self.saving_path + self.current_scan_file)
                        self.log_message("SpecMan4EPR: Analyze Done %s" %self.cur_exp.name, verbosity)
                        break        
        else:
            dset = self._create_fake_dataset()
        
        return super().acquire_dataset(dset)
    
    def tune_rectpulse(self,*,tp, LO, B, reptime,**kwargs):
        if SKIP_OPT_PULSE:
            self.pulse90 = RectPulse(tp=0.22, freq=0, flipangle=np.pi/2, scale=None)
            self.pulse180 = RectPulse(tp=0.55, freq=0, flipangle=np.pi/2, scale=None)
        else:
            self.pulse90 = RectPulse(tp=tp, freq=0, flipangle=np.pi/2, scale=None)
            pulse90_name = 'Observe'
            self.pulse90 = self.optimize_rectpulse(self.pulse90, self.t90, self.t180, self.t_nut, LO, B, self.reptime, self.awg_freq, pulse90_name, fast = True)
        
            self.pulse180 = RectPulse(tp=tp, freq=0, flipangle=np.pi/2, scale=None)
            pulse180_name = 'Refocus'
            self.pulse180 = self.optimize_rectpulse(self.pulse180, self.t90, self.t180, self.t_nut, LO, B, self.reptime, self.awg_freq, pulse180_name, fast = True)
        
        return self.pulse90, self.pulse180
        
    def optimize_rectpulse(self, pulse, t90, t180, t_nut, LO, B, reptime, modulation_freq, pulse_name, fast = False, **kwargs):
        '''
        optimize rectangular pulse
        Args:
            pulse: the pulse that will be optimized
            t90: pulse length in s
            t180: pulse length in s
            t_nut: pulse length in s
            LO: Synthesizer Frequency in Hz
            Reptime: Repeat time in s
            Modualation_freq: Modulation frequency in Hz
            pulse_type: Observe, Refocus or Pump
    
        '''
        # local file name is necessary
        fguid = 'AD_TunePulse_' + pulse_name + "_" +  datetime.datetime.now().strftime('%Y%m%d_%H%M')
        
        self.LO = LO * 1e9
        self.field = B
        if self.field_unit == 'T':
            self.field *= 1e-4
        hw_log.debug(f"Tuning {pulse.name} pulse")
        self.log_message("SpecMan4EPR: Setting experiment: Optimize pulse for %s " %pulse.name, verbosity)    
        if fast:
            if modulation_freq * 1e-6 != 500:
                print('Sequence on Hardware is not available because the modualtion frequency is not 500 MHz, current mod freq is %i MHz' %(modulation_freq*1e-6))
                self.send_message(".server.open = 'AutoDEER/AD_TunePulse_%s.tpl'" %pulse_name)
            else:
                self.send_message(".server.open = 'AutoDEER/AD_TunePulse_%s_fast.tpl'" %pulse_name)
        else:
            self.send_message(".server.open = 'AutoDEER/AD_TunePulse_%s.tpl'" %pulse_name)
        self.send_message(".spec.FLD.Field = %0.03f"%self.field)
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(self.reptime*1e6))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'" %(t90 * 1e9))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'" %(t180 * 1e9))
        self.send_message(".exp.expaxes.P.t_nut.string = '%i ns'" %(t_nut * 1e9))
        self.send_message(".exp.expaxes.P.f.string = '%i MHz'" %(modulation_freq * 1e-6))
        self.send_message(".exp.expaxes.P.ModulationFrequency.string = '%i MHz'" %(modulation_freq * 1e-6))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %self.amp180)
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %self.amp90)
        self.send_message(".server.COMPILE")
        time.sleep(15)
 
        self.send_message(".daemon.fguid = '%s'"%fguid)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s" %(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(self.reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(self.amp180), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(self.amp90), verbosity)
        self.log_message("SpecMan4EPR: amp_nut = %f" %(self.amp_nut), verbosity)
        self.log_message("SpecMan4EPR: f = %i MHz" %(modulation_freq * 1e-6))
        self.log_message("SpecMan4EPR: ModulationFrequency = %i MHz" %(modulation_freq * 1e-6))
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)

        while self.isrunning():
            continue
        data = self._create_dnpdata(self.saving_path + fguid + '.exp')
        dim = data.dims[0]
        data = data.abs
        # dnp.plt.figure()
        # dnp.fancy_plot(data)
        # dnp.plt.show()
        # data = dnp.autophase(data, dim = dim)
        # data = dnp.smooth(data, dim = dim)
        scale = data.argmax(dim).values

        if pulse_name == 'Observe': # 90 pulse
            self.amp90 = scale
            # pulse.tp.value = t90 * 1e9
        elif pulse_name == 'Refocus': # 180 pulse
            self.amp180 = scale
            # pulse.tp.value = t180 * 1e9
        elif pulse_name == 'Pump': # Pump pulse
            scale = data['amp_nut', (self.amp90, 1.)].argmax(dim).values
            self.amp_nut = scale
            # pulse.tp.value = t_nut * 1e9
        else:
            raise TypeError('Pulse Name is not acceptable')
        
        pulse.scale.value = scale
         
        if self.amp180 <= self.amp90:
            pulse = self.optimize_rectpulse(pulse, self.t90, self.t180, self.t_nut, LO, B, self.reptime, self.awg_freq, pulse_name, fast = fast)

        return pulse
    
    def tune_hspulse(self,*,tp, LO, B, reptime,**kwargs):

        return 'skip'

    def tune_pulse(self, pulse, mode, LO, B , reptime, shots=400):
        tp = pulse.tp.value # ns
        pulse_name = self.optimze_list.pop(0)
        if pulse_name == 'Observe':
            self.t90 = tp * 1e-9
            mod_freq = self.awg_freq
        elif pulse_name == 'Refocus':
            self.t180 = tp * 1e-9
            mod_freq = self.awg_freq
        elif pulse_name == "Pump":
            # self.t_nut = tp * 1e-9
            # mod_freq = self.awg_freq + (pulse.final_freq.value + pulse.init_freq.value) / 2 * 1e9 
            self.t_nut = 12 * 1e-9 # need to change later
            mod_freq = self.awg_freq - 90 * 1e6 
            
        if isinstance(pulse, RectPulse):
            pulse = self.optimize_rectpulse(pulse, self.t90, self.t180, self.t_nut, LO, B, self.reptime, mod_freq, pulse_name, fast = True)
        else:
            # need to change later
            pulse = self.optimize_rectpulse(pulse, self.t90, self.t180, self.t_nut, LO, B, self.reptime, mod_freq, pulse_name, fast = False)
        
        return pulse
            
    def isrunning(self) -> bool:
        '''
        4 - idle
        6 - run 
        '''

        check_string = self.send_message(".daemon.state")
        n = 0 # reading time
        # clean buffer
        while '=6' not in check_string and '=4' not in check_string and '=2' not in check_string:
            check_string += self.send_message(".daemon.state")
            time.sleep(0.5)
            if n >= 30:
                print('too much traffic in buffer: %i' %n)
                print(check_string)
                return False
            n+=1

        # ensure the experiment has been terminated appropriately
        while '=2' in self.send_message(".daemon.state"):
            time.sleep(0.5)
            continue

        if '=6' in check_string:
            return True
        elif '=4' in check_string:
            return False
        elif "=1" in check_string:
            raise SystemError('SpecMan has ERROR')

    def terminate(self) -> None:
        self.state = False
        self.log_message("SpecMan4EPR: Experiment Terminates", verbosity)
        self.send_message('.daemon.stop')
        time.sleep(1)
        self.fguid = self.current_scan_file.replace('.exp', '').replace('.d01', '')
        return super().terminate()
    
    def terminate_at(self, criterion, test_interval=2, keep_running=False, verbosity=0, autosave=True):
        '''
        Modified from super class
        
        '''
        test_interval_seconds = test_interval * 60
        condition = False
        last_scan = 0
        
        self.log_message('SpecMan4EPR: Monitoring the experiment...', verbosity) 
        self.log_message('SpecMan4EPR: Analyze data every %i s'%test_interval_seconds, verbosity)
        while not condition:

            if not self.isrunning():
                msg = "Experiments has finished before criteria met."
                # raise RuntimeError(msg)
                print(msg)
                return 

            start_time = time.time()
            data = self.acquire_dataset()
            if autosave:
                self.log.debug(f"Autosaving to {os.path.join(self.savefolder,self.savename)}")
                data.to_netcdf(os.path.join(self.savefolder,self.savename),engine='h5netcdf',invalid_netcdf=True)

            try:
                # nAvgs = data.num_scans.value
                nAvgs = data.attrs['nAvgs']

            except AttributeError or KeyError:
                self.log.warning("WARNING: Dataset missing number of averages(nAvgs)!")
                nAvgs = 1
            finally:
                if nAvgs < 1:
                    time.sleep(30)  # Replace with single scan time
                    continue
                elif nAvgs <= last_scan:
                    time.sleep(30)
                    continue    
            last_scan = nAvgs

            if verbosity > 0:
                print("Testing")

            if isinstance(criterion,list):
                conditions = [crit.test(data, verbosity) for crit in criterion]
                condition = any(conditions)
                
            else:
                condition = criterion.test(data, verbosity)
            if not condition:

                self.log_message('SpecMan4EPR: Condition Not Meet at average: %i'%nAvgs, verbosity)
                self.log_message('SpecMan4EPR: Waiting for next scan data', verbosity)

                end_time = time.time()
                if (end_time - start_time) < test_interval_seconds:
                    if verbosity > 0:
                        print("Sleeping")
                    time.sleep(test_interval_seconds - (end_time - start_time))
                
            else:
                if isinstance(criterion,list):
                    self.log_message('SpecMan4EPR: Condition Meet at average %i due to %s'%(nAvgs, [crit.name for crit in criterion if crit.test(data, verbosity)]), verbosity)
                else:
                    self.log_message('SpecMan4EPR: Condition Meet at average %i due to %s'%(nAvgs, criterion.name), verbosity)
        
        if isinstance(criterion,list):
            for i,crit in enumerate(criterion):
                if conditions[i]:
                    if callable(crit.end_signal):
                        crit.end_signal()
                
        else:
            if callable(criterion.end_signal):
                criterion.end_signal()
        
        self.terminate()
        pass
    
    def _run_fsweep(self, sequence: FieldSweepSequence):
        self.log_message("SpecMan4EPR: Setting experiment: Fieldsweep", verbosity)
        self.LO = sequence.LO.value * 1e9
        reptime = sequence.reptime.value * 1e-6
        self.ncyc = 1
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        pulse90 = sequence.pi2_pulse
        pulse180 = sequence.pi_pulse
        self.send_message(".server.open = 'AutoDEER/AD_Fieldsweep.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".exp.EXPAXES.X.Field.string = '%0.03f %s to %0.03f %s'"%(self.field - 0.007, self.field_unit, self.field + 0.013, self.field_unit))
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(reptime*1e6))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'" %(pulse90.tp.value))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'" %(pulse180.tp.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".server.COMPILE")
        time.sleep(1)
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        
        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s to %0.03f %s"%(self.field - 0.007, self.field_unit, self.field + 0.013, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(pulse90.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(pulse180.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: ncyc = %i" %(self.ncyc), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)
    
    def _run_reptimescan(self, sequence: ReptimeScan):
        self.log_message("SpecMan4EPR: Setting experiment: ShotRepetitionTime", verbosity)
        
        self.LO = sequence.LO.value * 1e9
        self.field = sequence.B.value
        if self.field_unit == 'T':
            self.field *= 1e-4
        
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        pulse90 = sequence.pi2_pulse
        pulse180 = sequence.pi_pulse
        self.send_message(".server.open = 'AutoDEER/AD_ShotRepetitionTime.tpl'")
        self.send_message(".spec.FLD.Field = %0.03f"%self.field)
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'" %(pulse90.tp.value))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'" %(pulse180.tp.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".server.COMPILE")
        time.sleep(15)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message('Set Field to %0.03f %s'%(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s" %(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(pulse90.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(pulse180.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: ncyc = %i" %(self.ncyc), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)
    
    def _run_respro(self, sequence: ResonatorProfileSequence):
        self.log_message("SpecMan4EPR: Setting experiment: ResonatorProfile", verbosity)
        self.field = sequence.B.value
        self.LO = sequence.LO.value * 1e9
        if self.field_unit == 'T':
            self.field *= 1e-4
        self.reptime = sequence.reptime.value * 1e-6
        Frequency, Field = self._calculate_nutation_bandwidth(self.LO, self.awg_freq, self.field, self.resonator_bandwidth)
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        pulse90 = sequence.pi2_pulse
        pulse180 = sequence.pi_pulse
        self.send_message(".server.open = 'AutoDEER/AD_ResonatorProfile.tpl'")
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".exp.expaxes.Y.Frequency.string = '%s'" %Frequency)
        self.send_message(".exp.expaxes.Y.Field.string = '%s'" %Field)
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(self.reptime*1e6))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'" %(pulse90.tp.value))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'" %(pulse180.tp.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".server.COMPILE")
        time.sleep(1)
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message("SpecMan4EPR: Frequency %s " %(Frequency), verbosity)
        self.log_message("SpecMan4EPR: Field = %s" %(Field), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(self.reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(pulse90.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(pulse180.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: ncyc = %i" %(self.ncyc), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)

    def _run_CP_relax(self, sequence: DEERSequence):
        self.log_message("SpecMan4EPR: Setting experiment: CarrPurcell", verbosity)
        
        self.LO = sequence.LO.value * 1e9
        self.reptime = sequence.reptime.value * 1e-6
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        self.field = sequence.B.value
        if self.field_unit == 'T':
            self.field *= 1e-4
        pulse90 = sequence.exc_pulse
        pulse180 = sequence.ref_pulse
        self.send_message(".server.open = 'AutoDEER/AD_CarrPurcell.tpl'")
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(self.reptime*1e6))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'" %(pulse90.tp.value))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'" %(pulse180.tp.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        self.send_message(".server.COMPILE")
        time.sleep(1)
        self.send_message(".spec.FLD.Field = %0.03f"%self.field)
        time.sleep(10)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s" %(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(self.reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(pulse90.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(pulse180.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: ncyc = %i" %(self.ncyc), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)

    def _run_T2_relax(self, sequence: T2RelaxationSequence):
        self.log_message("SpecMan4EPR: Setting experiment: HahnEchoDecay", verbosity)
        
        self.LO = sequence.LO.value * 1e9
        self.reptime = sequence.reptime.value * 1e-6
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        self.field = sequence.B.value
        if self.field_unit == 'T':
            self.field *= 1e-4
        pulse90 = sequence.pi2_pulse
        pulse180 = sequence.pi_pulse
        self.send_message(".server.open = 'AutoDEER/AD_HahnEchoDecay.tpl'")
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(self.reptime*1e6))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'" %(pulse90.tp.value))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'" %(pulse180.tp.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        self.send_message(".server.COMPILE")
        time.sleep(1)
        self.send_message(".spec.FLD.Field = %0.03f"%self.field)
        time.sleep(10)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s" %(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(self.reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(pulse90.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(pulse180.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: ncyc = %i" %(self.ncyc), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)

    def _run_2D_relax(self, sequence: RefocusedEcho2DSequence):
        self.log_message("SpecMan4EPR: Setting experiment: Refocused Echo", verbosity)
        
        self.LO = sequence.LO.value * 1e9
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        self.reptime = sequence.reptime.value * 1e-6
        self.field = sequence.B.value
        if self.field_unit == 'T':
            self.field *= 1e-4

        pulse90 = sequence.pi2_pulse
        pulse180 = sequence.pi_pulse
        self.send_message(".server.open = 'AutoDEER/AD_2DRefocusedEcho.tpl'")
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(self.reptime*1e6))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'" %(pulse90.tp.value))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'" %(pulse180.tp.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        self.send_message(".server.COMPILE")
        time.sleep(1)
        self.send_message(".spec.FLD.Field = %0.03f"%self.field)
        time.sleep(10)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s" %(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(self.reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(pulse90.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(pulse180.tp.value), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: ncyc = %i" %(self.ncyc), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)

    def _run_4pdeer(self, sequence: DEERSequence):
        self.log_message("SpecMan4EPR: Setting experiment: 4PulseDEER", verbosity)
        
        # force tau1 and tau2 to a value
        if self.tau1_fix:
            self.tau1 = self.guess_tau1
            sequence.tau1.value = ((self.tau1 * 1e9)//4 + 1) * 4 
        else:
            sequence.tau1.value = ((sequence.tau1.value)//4 + 1) * 4
            self.tau1 = sequence.tau1.value * 1e-9 

        if self.tau2_fix:
            self.tau2 = self.guess_tau2
            sequence.tau2.value = ((self.tau2 * 1e9)//4 + 1) * 4 
        else:
            sequence.tau2.value = ((sequence.tau2.value)//4 + 1) * 4
            self.tau2 = sequence.tau2.value * 1e-9
            # self.tau2 = self._check_tau2(self.tau2)
            # self.t_stop = self.tau2 - self.t180 - 1e-9
            # self.t_size = int((self.t_stop - self.t_start)/self.t_step) + 1
        
        self.t_start = -1 * self.tau1 + 40 * 1e-9
        t_step_list = [8, 16, 32, 64]
        t_step = 8
        for i in range(1, len(t_step_list)):
            if t_step_list[i] * 1e-9 * 130 + self.t_start - 1e-9 > self.tau2:
                t_step = t_step_list[i-1]
                break

        t_string = '%i ns step %i ns' %(self.t_start*1e9, t_step)
        # print(t_string)
        self.LO = sequence.LO.value * 1e9
        self.field = sequence.B.value
        if self.field_unit == 'T':
            self.field *= 1e-4
        
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        self.reptime = sequence.reptime.value * 1e-6
        
        pulse90 = sequence.exc_pulse
        pulse180 = sequence.ref_pulse
        pump_pulse = sequence.pump_pulse

        # determine pump pulse and observe pump
        # self.LO += sequence.exc_pulse.freq.value * 1e9 # change LO to observe, do not change awg frequency
        # f_eldor = self.awg_freq + (sequence.pump_pulse.freq.value - sequence.exc_pulse.freq.value) * 1e9 #change f_eldor to pump, the distance between pump and observe should be same 
        
        f_eldor = self.awg_freq + (-90)*1e6 # for now just change the pump pulse
        self.send_message(".server.open = 'AutoDEER/AD_4PulseDEER.tpl'")
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.X.size = %i" %self.t_size)
        self.send_message(".exp.EXPAXES.X.t.string = '%s'" %t_string)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(self.reptime*1e6))
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".exp.expaxes.P.tau1.string = '%0.01f ns'"%(self.tau1*1e9))
        self.send_message(".exp.expaxes.P.tau2.string = '%0.01f ns'"%(self.tau2*1e9))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'"%(self.t90*1e9))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'"%(self.t180*1e9))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".exp.expaxes.P.f_eldor.string = '%0.01f MHz'" %(f_eldor*1e-6))
        self.send_message(".exp.expaxes.P.t_eldor.string = '%0.01f ns'" %(self.t_nut*1e9))
        self.send_message(".exp.expaxes.P.amp_eldor.string = '%f'" %(pump_pulse.scale.value))
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        self.send_message(".server.COMPILE")
        time.sleep(1)
        self.send_message(".spec.FLD.Field = %0.03f"%self.field)
        time.sleep(10)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s" %(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(self.reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: tau1 = %0.01f ns" %(self.tau1 * 1e9), verbosity)
        self.log_message("SpecMan4EPR: tau2 = %0.01f ns" %(self.tau2 * 1e9), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(self.t90*1e9), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(self.t180*1e9), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: f_eldor = %0.01f MHz" %(f_eldor*1e-6), verbosity)
        self.log_message("SpecMan4EPR: t_eldor = %i ns" %(self.t_nut*1e9), verbosity)
        self.log_message("SpecMan4EPR: amp_eldor = %f" %(pump_pulse.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t step = %i ns" %(t_step), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)

    def _run_5pdeer(self, sequence: DEERSequence):
        self.log_message("SpecMan4EPR: Setting experiment: 5PulseDEER", verbosity)
        
        sequence.tau1.value = round(sequence.tau1.value, 1)
        sequence.tau2.value = round(sequence.tau2.value, 1)
        sequence.tau3.value = round(sequence.tau3.value, 1)

        self.tau1 = sequence.tau1.value * 1e-9 
        self.tau2 = sequence.tau2.value * 1e-9
        self.tau3 = sequence.tau3.value * 1e-9

        t_step_list = [8, 16, 32, 64]
        t_step = 8
        for i in range(1, len(t_step_list)):
            if t_step_list[i]* 241 + 100 - 1> (self.tau2 + self.tau1) * 1e9:
                t_step = t_step_list[i-1]
                break

        t_string = '%i ns step %i ns' %(100, t_step)

        self.LO = sequence.LO.value * 1e9
        self.field = sequence.B.value
        if self.field_unit == 'T':
            self.field *= 1e-4
        
        self.shots = sequence.shots.value
        self.scans = sequence.averages.value
        self.reptime = sequence.reptime.value * 1e-6
        
        pulse90 = sequence.exc_pulse
        pulse180 = sequence.ref_pulse
        pump_pulse = sequence.pump_pulse
        
        f_eldor = self.awg_freq + (-90)*1e6 # for now just change the pump pulse
        self.send_message(".server.open = 'AutoDEER/AD_5PulseDEER.tpl'")
        self.send_message(".exp.EXPAXES.I.reps = %i" %self.shots)
        self.send_message(".exp.EXPAXES.X.size = %i" %self.t_size)
        self.send_message(".exp.EXPAXES.X.t.string = '%s'" %t_string)
        self.send_message(".exp.EXPAXES.P.reps = %i" %self.scans)
        self.send_message(".exp.expaxes.P.RepTime.string = '%0.01f us'" %(self.reptime*1e6))
        self.send_message(".spec.BRIDGE.Frequency = %0.02f" %(self.LO))
        self.send_message(".exp.expaxes.P.tau1.string = '%0.01f ns'"%(self.tau1*1e9))
        self.send_message(".exp.expaxes.P.tau2.string = '%0.01f ns'"%(self.tau2*1e9))
        self.send_message(".exp.expaxes.P.tau3.string = '%0.01f ns'"%(self.tau3*1e9))
        self.send_message(".exp.expaxes.P.t90.string = '%i ns'"%(self.t90*1e9))
        self.send_message(".exp.expaxes.P.t180.string = '%i ns'"%(self.t180*1e9))
        self.send_message(".exp.expaxes.P.amp90.string = '%f'" %(pulse90.scale.value))
        self.send_message(".exp.expaxes.P.amp180.string = '%f'" %(pulse180.scale.value))
        self.send_message(".exp.expaxes.P.f_eldor.string = '%0.01f MHz'" %(f_eldor*1e-6))
        self.send_message(".exp.expaxes.P.t_eldor.string = '%0.01f ns'" %(self.t_nut*1e9))
        self.send_message(".exp.expaxes.P.amp_eldor.string = '%f'" %(pump_pulse.scale.value))
        self.send_message(".daemon.fguid = '%s'"%self.fguid)
        self.send_message(".server.COMPILE")
        time.sleep(1)
        self.send_message(".spec.FLD.Field = %0.03f"%self.field)
        time.sleep(10)
        if '0' in self.send_message('.daemon.state'):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")

        self.log_message("SpecMan4EPR: Frequency = %0.02f GHz" %(self.LO*1e-9), verbosity)
        self.log_message("SpecMan4EPR: Field = %0.03f %s" %(self.field, self.field_unit), verbosity)
        self.log_message("SpecMan4EPR: Reptime = %0.01f us" %(self.reptime*1e6), verbosity)
        self.log_message("SpecMan4EPR: shots = %i" %(self.shots), verbosity)
        self.log_message("SpecMan4EPR: scans = %i" %(self.scans), verbosity)
        self.log_message("SpecMan4EPR: tau1 = %0.01f ns" %(self.tau1 * 1e9), verbosity)
        self.log_message("SpecMan4EPR: tau2 = %0.01f ns" %(self.tau2 * 1e9), verbosity)
        self.log_message("SpecMan4EPR: tau3 = %0.01f ns" %(self.tau3 * 1e9), verbosity)
        self.log_message("SpecMan4EPR: t90 = %i ns" %(self.t90*1e9), verbosity)
        self.log_message("SpecMan4EPR: amp90 = %f" %(pulse90.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t180 = %i ns" %(self.t180*1e9), verbosity)
        self.log_message("SpecMan4EPR: amp180 = %f" %(pulse180.scale.value), verbosity)
        self.log_message("SpecMan4EPR: f_eldor = %0.01f MHz" %(f_eldor*1e-6), verbosity)
        self.log_message("SpecMan4EPR: t_eldor = %i ns" %(self.t_nut*1e9), verbosity)
        self.log_message("SpecMan4EPR: amp_eldor = %f" %(pump_pulse.scale.value), verbosity)
        self.log_message("SpecMan4EPR: t step = %i ns" %(t_step), verbosity)
        self.log_message("SpecMan4EPR: Experiment Launches", verbosity)

    def _connect_epr(self, address: str = 'localhost' , port_number: int = 8023) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (address, port_number)
        self.sock.connect(server_address)
        self.send_message(".server.message='SpecMan is controlled'")
        self.log_message('SpecMan4EPR: SpecMan is Controlled', verbosity)

    def send_message(self, message, convert:bool = True, request:bool = False): 
        if message[-1] != '\n':
            message += '\n'
        self.sock.send(message.encode('utf-8'))
        time.sleep(0.2)
        data = self.sock.recv(4096)

        if convert: 
            data = data.decode('utf-8')
            if request:
                data = data.split('=')[1]
                data = data.split('\r\n')[0]
                if data in ['True', 'False']:
                    return eval(data)
                else:
                    if '.' in data:
                        try:
                            data = float(data)
                            return data
                        except:
                            return data
                    else:
                        try: 
                            data = int(data)
                            return data
                        except:
                            return data
        return data
    
    def __del__(self):
        self.send_message(".server.opmode = 'Standby'")
        self.sock.close()

    def _create_dnpdata(self, filepath):
        data = dnp.load(filepath, autodetect_coords = True, autodetect_dims = True)        
        data_real = data['x',0].sum('x')
        data_imag = data['x',1].sum('x')
        data = data_real - 1j*data_imag
        return data

    def _create_dataset_from_b12t(self, filepath):
        data = self._create_dnpdata(filepath)      

        for dim in data.dims:
            if dim + '_unit' in data.attrs and data.attrs[dim + '_unit'] == 'T':
                data.coords[dim] *= 1e4 # convert unit to Gauss
                data.attrs[dim + '_unit'] = 'G'

        coord_factors = []
        default_labels = ['X','Y','Z','T']
        if isinstance(self.cur_exp, ReptimeScan):
            dims = ['reptime']
            coord_factors = [1e6]
        elif isinstance(self.cur_exp, ResonatorProfileSequence):
            dims = ['pulse0_tp', 'LO']
            coord_factors = [1e9, 1e-9]
        elif isinstance(self.cur_exp, DEERSequence):
            if self.cur_exp.t.is_static():
                dims = ['tau1', 'tau2', 'tau3', 'tau4']
            else:
                dims = ['t', 't2', 't3', 't4']
                print('here')
                if self.cur_exp.name == '4pDEER':
                    data.coords[data.dims[0]] += self.tau1
            coord_factors = [1e6, 1e6, 1e6, 1e6]
            dims = dims[:len(data.dims)]
            coord_factors = coord_factors[:len(data.dims)]
        elif isinstance(self.cur_exp, (RefocusedEcho2DSequence)):
            # if any([abs(x) < 0.002 for x in data.values[:,0]]): # the first point is not valid
            #     start = data.coords['tau1'][1]
            #     end = data.coords['tau1'][-1] + data.coords['tau1'][0]
            #     data = data['tau1',(start,end), 'tau2',(start,end)]
            data.reorder(['tau1', 'tau2'])
            tau1_size = len(data.coords['tau1'])
            tau2_size = len(data.coords['tau2'])
            new_tau2_size = tau1_size
            divider = tau2_size//tau1_size
            new_tau2_coords = np.resize(data.coords['tau2'], (new_tau2_size, divider))[:,-1]
            new_data_values = np.reshape(data.values, (tau1_size, new_tau2_size, divider))[:, :, -1]
            data.values = new_data_values
            data.coords['tau2'] = new_tau2_coords

            dims = ['tau1', 'tau2', 'tau3', 'tau4']
            coord_factors = [1e6, 1e6, 1e6, 1e6]
            dims = dims[:len(data.dims)]
            coord_factors = coord_factors[:len(data.dims)]
        elif isinstance(self.cur_exp, (T2RelaxationSequence)):
            coord_factors = [1e6, 1e6, 1e6, 1e6]
            dims = default_labels[:len(data.dims)]
            coord_factors = coord_factors[:len(data.dims)]
        else:
            dims = default_labels[:len(data.dims)]
        
        if isinstance(self.cur_exp, FieldSweepSequence):
            data.coords['Field'] *= 1
        
        coords = data.coords.coords
        if coord_factors:
            for index, dim in enumerate(dims):
                coords[index] *= coord_factors[index]

        attrs = data.attrs
        attrs['LO'] = self.LO * 1e-9 # GHz
        attrs['shots'] = int(self.shots)
        attrs['nAvgs'] = int(data.dnplab_attrs.get('scans')) if data.dnplab_attrs.get('scans') is not None else 1
        attrs['nPcyc'] = int(self.ncyc)
        attrs['reptime'] = self.reptime * 1e6 # us
        attrs ={**attrs, **get_all_fixed_param(self.cur_exp)}
        return xr.DataArray(data.values, dims=dims, coords=coords, attrs=attrs)
    
    def _create_fake_dataset(self):

        if isinstance(self.cur_exp, DEERSequence):
            if self.cur_exp.t.is_static():
                axes, data = _simulate_CP(self.cur_exp)
            else:
                axes, data =_simulate_deer(self.cur_exp)
        elif isinstance(self.cur_exp, FieldSweepSequence):
            axes, data = _simulate_field_sweep(self.cur_exp)
        elif isinstance(self.cur_exp,ResonatorProfileSequence):
            mode = lambda x: lorenz_fcn(x, self.fc, self.fc/self.Q)
            x = np.linspace(self.Bridge['Min Freq'],self.Bridge['Max Freq'])
            scale = 75/mode(x).max()
            axes, data = _similate_respro(self.cur_exp, lambda x: lorenz_fcn(x, self.fc, self.fc/self.Q) * scale)
        elif isinstance(self.cur_exp, ReptimeScan):
            axes, data = _simulate_reptimescan(self.cur_exp)
        elif isinstance(self.cur_exp,T2RelaxationSequence):
            axes, data = _simulate_T2(self.cur_exp, 0.15)
        elif isinstance(self.cur_exp,RefocusedEcho2DSequence):
            axes, data = _simulate_2D_relax(self.cur_exp)
        else:
            raise NotImplementedError("Sequence not implemented")

        data = add_noise(data, 1/(75))
        scan_num = self.cur_exp.averages.value
        dset = create_dataset_from_sequence(data,self.cur_exp)
        dset.attrs['nAvgs'] = int(scan_num)

        return dset
        
    def _calculate_nutation_bandwidth(self, freq_bridge, freq_awg, field, freq_width):
        '''
        Calcaute Parameters for nutation bandwidth experiment
        Args: 
            freq_bridge: bridge frequency in Hz
            freq_awg: modulation frequency in Hz
            field: the center field in G
            freq_width: the bandwidth of resonator in Hz

        Returns:
            LO, Sweep strings for nutation bandwidth experiment
        
        '''

        # convert to GHz
        freq_bridge /= 1e9 
        freq_awg /= 1e9
        freq_width /= 1e9
        freq = freq_bridge + freq_awg
        field_to_freq = freq / field 

        field_sweep_low = (freq_bridge + freq_awg - freq_width/2.) / field_to_freq
        field_sweep_high = (freq_bridge + freq_awg + freq_width/2.) / field_to_freq

        freq_sweep_low = freq_bridge - freq_width/2.
        freq_sweep_high = freq_bridge + freq_width/2.

        Frequency = '%0.04f GHz to %0.04f GHz'%(freq_sweep_low, freq_sweep_high)
        Field = '%0.04f %s to %0.04f %s'%(field_sweep_low,self.field_unit,field_sweep_high,self.field_unit)

        return Frequency, Field

    def _calculate_observe_field(self, B, g, observe_freq):
        return B + observe_freq/g
    
    def _calculate_tau2(self, data):
        tau2s = []
        tau2_errs = []
        for t in data.coords['tau1']:
            temp = data['tau1', t].sum('tau1').abs
            temp /= np.max(temp)
            try:
                fit = dnp.fit(dnp.relaxation.t2, temp, dim = 'tau2', p0 = (1.,1.0e-6,1))
                tau2s.append(fit['popt'].values[1])
                tau2_errs.append(fit['err'].values[1])
                if len(tau2s) == 5:
                    break
            except:
                tau2s.append(99)
                tau2_errs.append(99)
                continue
        
        if all([x==99 for x in tau2s]):
            print('cannot find tau2')
            return None
        
        tau2s_diff_from_guess = [np.abs(x - self.guess_tau2) for x in tau2s] # find the closest value to initial guess
        tau2 = tau2s[np.argmin(tau2s_diff_from_guess)]
        if tau2 <= self.t_stop + self.t180:
            tau2 = self.t_stop + self.t180 + 1e-9 # prevent overlapped pulse
        return tau2
    
    def _check_tau2(self, tau2):
        if tau2 <= self.t_stop + self.t180:
            self.log_message("change tau2 from %0.02f us to %0.02f to prevent overlapping pulse" %(tau2*10e6, self.t_stop + self.t180 + 1e-9), verbosity)
            tau2 = self.t_stop + self.t180 + 1e-9 # prevent overlapped pulse
        return tau2
    
    def verify_field(self, target, maximum = 300):
        current_field = round(self.send_message(".spec.FLD.Field.value", request = True), 2)
        n = 0
        while current_field != target:
            wait(10) # wait field to settle
            current_field = round(self.send_message(".spec.FLD.Field.value", request = True), 2)
            n += 1
            if n > maximum: # 5 mins
                return False
        return True
    
    def log_message(self, message, verbosity: int = 0):
        if verbosity >= 1: 
            hw_log.info(message)
        
        if verbosity >= 2:
            self.send_message(".server.message='%s'" %message)
        
        if verbosity == 3:
            print(message)


# def func(x, a, b, e):
#     return a*np.exp(-b*x**e)

# def _simulate_CP(sequence):

#     if isinstance(sequence, DEERSequence):
#         xaxis = sequence.tau2.value
#     elif isinstance(sequence, CarrPurcellSequence):
#         xaxis = sequence.step.value
    
#     data = func(xaxis,1,2e-6,1.8)
#     data = add_phaseshift(data, 0.05)
#     return xaxis, data

# def _simulate_T2(sequence,ESEEM_depth):
#     xaxis = sequence.tau.value
#     data = func(xaxis,1,10e-6,1.6)
#     data = add_phaseshift(data, 0.05)
#     if ESEEM_depth != 0:
#         data *= _gen_ESEEM(xaxis, 7.842, ESEEM_depth)
#     return xaxis, data

def lorenz_fcn(x, centre, sigma):
    y = (0.5*sigma)/((x-centre)**2 + (0.5*sigma)**2)
    return y   

def wait(time_in_second):
    '''
    Count down the time without sleep
    '''
    start_time = time.time()
    while True:
        if time.time() - start_time > time_in_second:
            break