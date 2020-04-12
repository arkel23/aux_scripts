#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from bisect import bisect_left
plt.ioff()

EPSILON = 0.01 # s = 10 ms * 50 diff between calc time and time truth, 10 ms for calc time and time frame
LOW = 22

def metrics(time_truth, hr_truth, time_calc, hr_calc, calc_info, staticFlag, printFlag):
    N = len(hr_truth)
    M = len(hr_calc)
    if (N == M):

        #root mean squared error
        rmse_top_root = np.sum(np.power(hr_truth - hr_calc,2))
        rmse = np.sqrt(rmse_top_root/N)
        #mean absolute error
        mae = np.sum(np.abs(hr_truth - hr_calc))/N
        #mean absolute percentage error
        mape = (100/N) * np.sum(np.abs((hr_truth - hr_calc)/hr_truth))
        
        if (staticFlag == False):
            mean_hr_truth = np.mean(hr_truth)
            mean_hr_calc = np.mean(hr_calc)
            #print(mean_hr_truth, mean_hr_calc)
            #covariance
            #definition from vision-based heart rate estimation wang 2019
            #pear_corr_top = np.sum(hr_truth - mean_hr_truth)*np.sum(hr_calc - mean_hr_calc)
            #definition from wikipedia
            pear_corr_top = np.sum((hr_truth - mean_hr_truth)*(hr_calc - mean_hr_calc))
            #standard deviations
            std_dev_truth = np.sqrt(np.sum(np.power(hr_truth - mean_hr_truth,2)))
            std_dev_calc = np.sqrt(np.sum(np.power(hr_calc - mean_hr_calc,2)))
            #print(std_dev_truth, std_dev_calc)
            pear_corr = pear_corr_top/(std_dev_truth*std_dev_calc)
            
        if (printFlag):
            print('Results for window size: ', calc_info)
            print('RMSE: ', rmse)
            print('MAE: ', mae)
            print('MAPE: ', mape)
            if (staticFlag == False):
                print('Pearsons: ', pear_corr)

def plots_individual(time_truth, hr_truth, time_calc, hr_calc, calc_info, staticFlag):
    
    fig, axs = plt.subplots(2, 1)
    #fig.set_size_inches((640/10, 480/10)) 
    fig.subplots_adjust(hspace=0.5)
                       
    fig.suptitle('Calculated file info: {}'.format(calc_info))
    axs[0].plot(time_truth, hr_truth)
    axs[1].plot(time_calc, hr_calc)
    if (staticFlag == False):
        axs[0].set_ylim(np.min(hr_truth)-np.std(hr_truth), np.max(hr_truth)+np.std(hr_truth))
    axs[1].set_ylim(np.min(hr_calc)-np.std(hr_calc), np.max(hr_calc)+np.std(hr_calc))
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Heart rate (BPM)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Heart rate (BPM)')
    axs[0].set_title('Ground truth values')
    axs[1].set_title('Calculated values')
    axs[0].grid(True)
    axs[1].grid(True)
    fig.canvas.draw()
    out_dir = './results/'
    os.makedirs(out_dir, exist_ok=True)
    if (staticFlag == False):
        plt.savefig(os.path.join(out_dir,'{}.png'.format(calc_info)))
    else:
        plt.savefig(os.path.join(out_dir,'{}_staticBPM.png'.format(calc_info)))
        
def get_closests(df, val):
    lower_idx = bisect_left(df, val) #val is in the list
    return lower_idx

def undersampling(time_calc, hr_calc, time_truth, hr_truth, len_calc_orig, len_true_orig):
    if len_true_orig > len_calc_orig:
        time_ss = np.reshape(time_calc, (-1, 1))
        hr_ss = np.reshape(hr_calc, (-1, 1))
        time_us = np.zeros((len_calc, 1))
        hr_us = np.zeros((len_calc, 1))
        len_same = len_calc_orig
        len_bigger = len_true_orig

        j = 0
        for i in range(len_bigger):
            if (abs(time_calc[j] - time_truth[i]) < EPSILON*50):
                time_us[j] = time_truth[i]
                hr_us[j] = hr_truth[i]
                j += 1
                print(j, time_us[j], hr_us[j], time_ss[j], hr_ss[j])
                if (j == len_same):
                    break;
    else:
        time_ss = np.reshape(time_truth, (-1, 1))
        hr_ss = np.reshape(hr_truth, (-1, 1))
        time_us = np.zeros((len_true_orig, 1))
        hr_us = np.zeros((len_true_orig, 1))
        len_same = len_true_orig
        len_bigger = len_calc_orig

        j = 0
        for i in range(len_bigger):
            print(time_calc[i], hr_calc[i], time_truth[j], hr_truth[j])
            if (abs(time_truth[j] - time_calc[i]) < EPSILON*50):
                time_us[j] = time_calc[i]
                hr_us[j] = hr_calc[i]
                print(j, time_us[j], hr_us[j], time_ss[j], hr_ss[j])
                j += 1
                if (j == len_same):
                    break;

    return time_us, hr_us, time_ss, hr_ss, len_same, len_bigger

def load_HR(path_calc_str, path_truth_str, low_time, printFlag):

    # Problems that I see with reading these CSV files is that in average they have around 2 K something samples, which means
    # a sampling rate of 30 Hz, pretty similar to the fps but it doesn't exactly match the frames
    
    # read the ground_truth and the calculated values and store them
    path_calc = Path(path_calc_str)
    df_calc = pd.read_csv(path_calc, delim_whitespace=True)
    df_calc.columns = df_calc.columns.str.lower() #MAKE SURE ALL BPMs are written in LOWER CASE
    time_calc = np.array(df_calc['time']/1000) #ms to seconds
    hr_calc = np.array(df_calc['bpm'])

    base=os.path.basename(os.path.normpath(path_calc))
    calc_name = os.path.splitext(base)[0]

    staticBPMflag = False

    if (path_truth_str):
        path_truth =  Path(path_truth_str).resolve()
        df_truth = pd.read_csv(path_truth, header=None, delim_whitespace=True)
        #print(df_truth)
        #print(df_truth.head())
        #print(df_truth.shape)
        hr_truth = np.array(df_truth.iloc[1])
        time_truth = np.array(df_truth.iloc[0])
    else:
        staticBPM = input('No ground truth file found. Please input static BPM: ')
        time_truth = np.copy(time_calc)
        hr_truth = np.ones((hr_calc.shape))*int(staticBPM)
        staticBPMflag = True
        
    print(hr_truth.shape, time_truth.shape, hr_calc.shape, time_calc.shape)
    print(hr_truth.dtype, time_truth.dtype, hr_calc.dtype, time_calc.dtype)
    print(time_truth[0], time_calc[0], time_truth[-1], time_calc[-1])

    #section the dataframes to only include values between
    # low_time s and high_time s
    high_time = time_calc[-60] #a few frames before the end of the video
    low_slice_calc = get_closests(time_calc, low_time)
    high_slice_calc = get_closests(time_calc, high_time)           
    low_slice_true = get_closests(time_truth, low_time)
    high_slice_true = get_closests(time_truth, high_time)
    print(low_slice_calc, high_slice_calc, low_slice_true, high_slice_true)

    hr_calc = hr_calc[low_slice_calc:high_slice_calc]
    time_calc = time_calc[low_slice_calc:high_slice_calc]
    hr_truth = hr_truth[low_slice_true:high_slice_true]
    time_truth = time_truth[low_slice_true:high_slice_true]

    print(hr_truth.shape, time_truth.shape, hr_calc.shape, time_calc.shape)
    print(time_truth[0], time_calc[0], time_truth[-1], time_calc[-1])

    len_calc_orig = len(hr_calc)
    len_true_orig = len(hr_truth)

    time_us, hr_us, time_ss, hr_ss, len_same, len_bigger = undersampling(time_calc, hr_calc, time_truth, hr_truth, len_calc_orig, len_true_orig)
    
    print(hr_us.shape, time_us.shape, hr_ss.shape, time_ss.shape)
    print(time_us[0], time_ss[0], time_us[-1], time_ss[-1])

    plots_individual(time_us, hr_us, time_ss, hr_ss, calc_name, staticBPMflag)
    metrics(time_us, hr_us, time_ss, hr_ss, calc_name, staticBPMflag, printFlag)

def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-low', dest='low', default=LOW, type=int, help='Time to begin analysis. Must be higher than window size.')
    parser.add_argument('-calc', required=True, dest='calc', type=str, help='Absolute path to calculated HR.')
    parser.add_argument('-gt', dest='gt', type=str, help='''Absolute path to ground truth file.
    If not specified program will ask during runtime for user to input an static BPM.''')
    parser.add_argument('-print', dest='printFlag', default=True, type=bool, help='''Print accuracy metrics or not. 
    True or False. Default is True.''')
    args = parser.parse_args()
    
    load_HR(args.calc, args.gt, args.low, args.printFlag)

main()

# %%
