'''
Convert_data

This module contains converters to easily convert data between multiple formats
The formats used within reliability are:
FR - failures, right censored
FNRN - failures, number of failures, right censored, number of right censored
XCN - event time, censoring code, number of events

The following data converters are available in this module:
FR_to_FNRN
FR_to_XCN
FNRN_to_FR
FNRN_to_XCN
XCN_to_FR
XCN_to_FNRN
xlsx_to_FR
xlsx_to_FNRN
xlsx_to_XCN

Note that each format has an acceptable reduced form where the omitted detail is assumed absent (i.e. no right censored data in the case of FR and FNRN)
or as all single events (i.e. all with a quantity of 1 in the case of XCN).
For example:
FR format may just be F if there is no right censored data
FNRN may be just FN if there is no right censored data. FNRN may not be just F as this is the same as F from FR format.
XCN may be just XC if there are no grouped values (ie. every event is assumed to have a quantity of 1). XCN may not be just X as this is the same as F from FR format.
'''

import numpy as np
import pandas as pd
from reliability.Utils import colorprint, write_df_to_xlsx, removeNaNs


class xlsx_to_XCN:
    '''
    xlsx_to_XCN data format converter

    Inputs:
    path - the filepath for the xlsx file. Note that you must prefix this with r to specify it as raw text. eg. path=r'.../documents/myfile.xlsx'
    censor_code_in_xlsx - specify the censor code you have used if it does not appear in the defaults (see below).
        *default censor codes that will be recognised (not case sensitive): 'R', 'RC', 'RIGHT CENS', 'RIGHT CENSORED', 'C', 'CENSORED', 'CENS', 'S', 'SUSP', 'SUSPENSION', 'SUSPENDED', 'UF', 'UNFAILED', 'UNFAIL', 'NF', 'NO FAIL', 'NO FAILURE', 'NOT FAILED', 1
    failure_code_in_xlsx - specify the failure code you have used if it does not appear in the defaults (see below).
        *default failure codes that will be recognised (not case sensitive): 'F', 'FAIL', 'FAILED', 0
    censor_code_in_XCN - specify the censor code to be used in XCN format. Default is 'C'
    failure_code_in_XCN - specify the failure code to be used in XCN format. Default is 'F'

    Output:
    X - event time
    C - censor code
    N - number of events at each event time

    Methods:
    print() - this will print a dataframe of the data in XCN format to the console
    write_to_xlsx() - this will export the data in XCN format to an xlsx file at the specified path.

    Example usage:
    XCN = xlsx_to_XCN(path=r'C:\...\Desktop\mydata.xlsx')

    For further usage examples, see the examples provided for FR_to_XCN or FNRN_to_XCN

    Note that the function is expecting the xlsx file to have columns in XCN format.
    If they are in another format (FR, FNRN) then you will need to use the appropriate function for that format.
    A reduced form (XC) is accepted and all values will be assumed to have a quantity (N) of 1.

    '''

    def __init__(self, path, censor_code_in_xlsx=None, failure_code_in_xlsx=None, censor_code_in_XCN='C', failure_code_in_XCN='F', **kwargs):
        df = pd.read_excel(io=path, **kwargs)
        cols = df.columns
        X = df[cols[0]].to_numpy()
        X = np.array(removeNaNs(list(X)))
        C0 = df[cols[1]].to_numpy()
        C0 = removeNaNs(C0)
        C_upper = []
        for item in C0:
            if type(item) in [str, np.str_]:
                C_upper.append(item.upper())  # for strings
            else:
                C_upper.append(item)  # for numbers
        C_unique = np.unique(C_upper)
        if len(C_unique) > 2:
            error_str = str('xlsx_to_XCN assumes the second column is C (censoring code). A maximum of 2 unique censoring codes are allowed. Within this column there were ' + str(len(C_unique)) + ' unique values: ' + str(C_unique))
            raise ValueError(error_str)
        C_out = []
        if type(failure_code_in_xlsx) in [str, np.str_]:  # need to upper() the input since we are comparing with C_upper
            failure_code_in_xlsx = failure_code_in_xlsx.upper()
        if type(censor_code_in_xlsx) in [str, np.str_]:
            censor_code_in_xlsx = censor_code_in_xlsx.upper()

        for item in C_upper:
            if item == failure_code_in_xlsx:
                C_out.append(failure_code_in_XCN)
            elif item == censor_code_in_xlsx:
                C_out.append(censor_code_in_XCN)
            elif item in ['F', 'FAIL', 'FAILED', 0]:
                C_out.append(failure_code_in_XCN)
            elif item in ['R', 'RC', 'RIGHT CENS', 'RIGHT CENSORED', 'C', 'CENSORED', 'CENS', 'S', 'SUSP', 'SUSPENSION', 'SUSPENDED', 'UF', 'UNFAILED', 'UNFAIL', 'NF', 'NO FAIL', 'NO FAILURE', 'NOT FAILED', 1]:
                C_out.append(censor_code_in_XCN)
            else:
                raise ValueError('Unrecognised value in the second column of the xlsx file. xlsx_to_XCN assumes the second column is C (censoring code). Common values are used as defaults but the xlsx file contained unrecognised values. You can fix this by specifying the arguments censor_code_in_xlsx  and failure_code_in_xlsx.')
        C = np.array(C_out)

        if len(cols) > 2:
            N = df[cols[2]].to_numpy()
            N = removeNaNs(N)
        else:
            N = np.ones_like(X)  # if N is missing then it is assumed as all ones
        if len(cols) > 3:
            colorprint("WARNING: xlsx_to_XCN assumes the first three columns in the excel file are being used for 'X' (event times), 'C' (censoring codes), 'N' (number of items at each event time). All other columns have been ignored", text_color='red')
        if len(X) != len(C) or len(X) != len(N):
            raise ValueError('The lengths of the first 3 columns in the xlsx file do not match. This may be because some data is missing.')

        FR = XCN_to_FR(X=X, C=C, N=N)  # we do this seeming redundant conversion to combine any duplicates from FNRN which were not correctly summarized in the input data
        XCN = FR_to_XCN(failures=FR.failures, right_censored=FR.right_censored)
        self.X = XCN.X
        self.C = XCN.C
        self.N = XCN.N
        Data = {'event time': self.X, 'censor code': self.C, 'number of events': self.N}
        self.__df = pd.DataFrame(data=Data, columns=['event time', 'censor code', 'number of events'])

    def print(self):
        colorprint('Data (XCN format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class xlsx_to_FR:
    '''
    xlsx_to_FR data format converter

    Inputs:
    path - the filepath for the xlsx file. Note that you must prefix this with r to specify it as raw text. eg. path=r'.../documents/myfile.xlsx'

    Output:
    failures
    right_censored

    Methods:
    print() - this will print a dataframe of the data in FR format to the console
    write_to_xlsx() - this will export the data in FR format to an xlsx file at the specified path.

    Example usage:
    FR = xlsx_to_FR(path=r'C:\...\Desktop\mydata.xlsx')

    For further usage examples, see the examples provided for XCN_to_FR or FNRN_to_FR

    Note that the function is expecting the xlsx file to have columns in FR format.
    If they are in another format (XCN, FNRN) then you will need to use the appropriate function for that format.
    A reduced form (F) is accepted and all values will be assumed to be failures.

    '''

    def __init__(self, path, **kwargs):
        df = pd.read_excel(io=path, **kwargs)
        cols = df.columns
        failures = df[cols[0]].to_numpy()
        self.failures = removeNaNs(failures)
        if len(cols) > 1:
            right_censored = df[cols[1]].to_numpy()
            self.right_censored = removeNaNs(right_censored)
            f, rc = list(self.failures), list(self.right_censored)
            len_f, len_rc = len(f), len(rc)
            max_len = max(len_f, len_rc)
            if not max_len == len_f:
                f.extend([''] * (max_len - len_f))
            if not max_len == len_rc:
                rc.extend([''] * (max_len - len_rc))
            Data = {'failures': f, 'right censored': rc}
            self.__df = pd.DataFrame(Data, columns=['failures', 'right censored'])
        else:
            self.right_censored = None
            Data = {'failures': self.failures}
            self.__df = pd.DataFrame(Data, columns=['failures'])

        if len(cols) > 2:
            colorprint("WARNING: xlsx_to_FR assumes the first two columns in the excel file are 'failures' and 'right censored'. All other columns have been ignored", text_color='red')

    def print(self):
        colorprint('Data (FR format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class xlsx_to_FNRN:
    '''
    xlsx_to_FNRN data format converter

    Inputs:
    path - the filepath for the xlsx file. Note that you must prefix this with r to specify it as raw text. eg. path=r'.../documents/myfile.xlsx'

    Output:
    failures
    num_failures
    right_censored
    num_right_censored

    Methods:
    print() - this will print a dataframe of the data in FNRN format to the console
    write_to_xlsx() - this will export the data in FNRN format to an xlsx file at the specified path.

    Example usage:
    FNRN = xlsx_to_FNRN(path=r'C:\...\Desktop\mydata.xlsx')

    For further usage examples, see the examples provided for XCN_to_FNRN or FR_to_FNRN

    Note that the function is expecting the xlsx file to have columns in FNRN format.
    If they are in another format (FR, XCN) then you will need to use the appropriate function for that format.
    A reduced form (FN) is accepted and all values will be assumed to be failures.

    '''

    def __init__(self, path, **kwargs):
        df = pd.read_excel(io=path, **kwargs)
        cols = df.columns
        failures = df[cols[0]].to_numpy()
        num_failures = df[cols[1]].to_numpy()
        failures = removeNaNs(failures)
        num_failures = removeNaNs(num_failures)
        if len(failures) != len(num_failures):
            raise ValueError("xlsx_to_FNRN assumes the first and second columns in the excel file are 'failures' and 'number of failures'. These must be the same length.")
        if len(cols) == 2:
            right_censored = None
            num_right_censored = None
        else:
            right_censored = df[cols[2]].to_numpy()
            num_right_censored = df[cols[3]].to_numpy()
            right_censored = removeNaNs(right_censored)
            num_right_censored = removeNaNs(num_right_censored)
            if len(right_censored) != len(num_right_censored):
                raise ValueError("xlsx_to_FNRN assumes the third and fourth columns in the excel file are 'right censored' and 'number of right censored'. These must be the same length.")
        if len(cols) > 4:
            colorprint("WARNING: xlsx_to_FNRN assumes the first four columns in the excel file are 'failures', 'number of failures', 'right censored', 'number of right censored'. All other columns have been ignored", text_color='red')

        FR = FNRN_to_FR(failures=failures, num_failures=num_failures, right_censored=right_censored, num_right_censored=num_right_censored)
        FNRN = FR_to_FNRN(failures=FR.failures, right_censored=FR.right_censored)  # we do this seeming redundant conversion to combine any duplicates from FNRN which were not correctly summarized in the input data
        self.failures = FNRN.failures
        self.num_failures = FNRN.num_failures
        self.right_censored = FNRN.right_censored
        self.num_right_censored = FNRN.num_right_censored

        # make the dataframe for printing and writing to excel
        if self.right_censored is not None:
            f, nf, rc, nrc = list(self.failures), list(self.num_failures), list(self.right_censored), list(self.num_right_censored)
            len_f, len_rc = len(f), len(rc)
            max_len = max(len_f, len_rc)
            if not max_len == len_f:
                f.extend([''] * (max_len - len_f))
                nf.extend([''] * (max_len - len_f))
            if not max_len == len_rc:
                rc.extend([''] * (max_len - len_rc))
                nrc.extend([''] * (max_len - len_rc))
            Data = {'failures': f, 'number of failures': nf, 'right censored': rc, 'number of right censored': nrc}
            self.__df = pd.DataFrame(Data, columns=['failures', 'number of failures', 'right censored', 'number of right censored'])
        else:
            Data = {'failures': self.failures, 'number of failures': self.num_failures}
            self.__df = pd.DataFrame(Data, columns=['failures', 'number of failures'])

    def print(self):
        colorprint('Data (FNRN format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class XCN_to_FNRN:
    '''
    XCN_to_FNRN data format converter

    Inputs:
    X - the failure or right_censored time. This must be an array or list.
    C -  the censoring code for each X. This must be an array or list. Defaults are recognised from the lists shown below.
    N - the quantity for each X. This must be an array or list. Optional Input. If omitted all items are assumed to have quantity (N) of 1.
    censor_code - specify the censor code you have used if it does not appear in the defaults (see below). Optional input.
        * default censor codes that will be recognised (not case sensitive): 'R', 'RC', 'RIGHT CENS', 'RIGHT CENSORED', 'C', 'CENSORED', 'CENS', 'S', 'SUSP', 'SUSPENSION', 'SUSPENDED', 'UF', 'UNFAILED', 'UNFAIL', 'NF', 'NO FAIL', 'NO FAILURE', 'NOT FAILED', 1
    failure_code - specify the failure code you have used if it does not appear in the defaults (see below). Optional Input.
        * default failure codes that will be recognised (not case sensitive): 'F', 'FAIL', 'FAILED', 0

    Output:
    failures
    num_failures
    right_censored
    num_right_censored

    Methods:
    print() - this will print a dataframe of the data in FNRN format to the console
    write_to_xlsx() - this will export the data in FNRN format to an xlsx file at the specified path.

    Example usage:
    FNRN = XCN_to_FNRN(X=[1,2,3,7,8,9], C=['f','f','f','c','c','c'], N=[1,2,2,3,2,1])
    print(FNRN.failures)
       >>> [1 2 3]
    print(FNRN.num_failures)
       >>> [1 2 2]
    print(FNRN.right_censored)
       >>> [7 8 9]
    print(FNRN.num_right_censored)
       >>> [3 2 1]
    FNRN.print()
       >>> Data (FNRN format)
           failures  number of failures  right censored  number of right censored
                  1                   1               7                         3
                  2                   2               8                         2
                  3                   2               9                         1

    '''

    def __init__(self, X, C, N=None, censor_code=None, failure_code=None):
        FR = XCN_to_FR(X=X, C=C, N=N, censor_code=censor_code, failure_code=failure_code)
        FNRN = FR_to_FNRN(failures=FR.failures, right_censored=FR.right_censored)
        self.failures = FNRN.failures
        self.num_failures = FNRN.num_failures
        self.right_censored = FNRN.right_censored
        self.num_right_censored = FNRN.num_right_censored
        # make the dataframe for printing and writing to excel
        if self.right_censored is not None:
            f, nf, rc, nrc = list(self.failures), list(self.num_failures), list(self.right_censored), list(self.num_right_censored)
            len_f, len_rc = len(f), len(rc)
            max_len = max(len_f, len_rc)
            if not max_len == len_f:
                f.extend([''] * (max_len - len_f))
                nf.extend([''] * (max_len - len_f))
            if not max_len == len_rc:
                rc.extend([''] * (max_len - len_rc))
                nrc.extend([''] * (max_len - len_rc))
            Data = {'failures': f, 'number of failures': nf, 'right censored': rc, 'number of right censored': nrc}
            self.__df = pd.DataFrame(Data, columns=['failures', 'number of failures', 'right censored', 'number of right censored'])
        else:
            Data = {'failures': self.failures, 'number of failures': self.num_failures}
            self.__df = pd.DataFrame(Data, columns=['failures', 'number of failures'])

    def print(self):
        colorprint('Data (FNRN format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class XCN_to_FR:
    '''
    XCN_to_FR data format converter

    Inputs:
    X - the failure or right_censored time. This must be an array or list.
    C -  the censoring code for each X. This must be an array or list. Defaults are recognised from the lists shown below.
    N - the quantity for each X. This must be an array or list. Optional Input. If omitted all items are assumed to have quantity (N) of 1.
    censor_code - specify the censor code you have used if it does not appear in the defaults (see below). Optional input.
        * default censor codes that will be recognised (not case sensitive): 'R', 'RC', 'RIGHT CENS', 'RIGHT CENSORED', 'C', 'CENSORED', 'CENS', 'S', 'SUSP', 'SUSPENSION', 'SUSPENDED', 'UF', 'UNFAILED', 'UNFAIL', 'NF', 'NO FAIL', 'NO FAILURE', 'NOT FAILED', 1
    failure_code - specify the failure code you have used if it does not appear in the defaults (see below). Optional Input.
        * default failure codes that will be recognised (not case sensitive): 'F', 'FAIL', 'FAILED', 0

    Output:
    failures
    right_censored

    Methods:
    print() - this will print a dataframe of the data in FR format to the console
    write_to_xlsx() - this will export the data in FR format to an xlsx file at the specified path.

    Example usage:
    FR = XCN_to_FR(X=[1,2,3,7,8,9], C=['f','f','f','c','c','c'], N=[1,2,2,3,2,1])
    print(FR.failures)
       >>> [1 2 2 3 3]
    print(FR.right_censored)
       >>> [7 7 7 8 8 9]
    FR.print()
       >>> Data (FR format)
           failures  right censored
                  1               7
                  2               7
                  2               7
                  3               8
                  3               8
                                  9

    '''

    def __init__(self, X, C, N=None, censor_code=None, failure_code=None):
        if type(N) == type(None):
            N = np.ones_like(X)  # assume a quantity of 1 if not specified
        if type(X) not in [list, np.ndarray]:
            raise ValueError('X must be a list or array.')
        if type(C) not in [list, np.ndarray]:
            raise ValueError('C must be a list or array.')
        if type(N) not in [list, np.ndarray]:
            raise ValueError('N must be a list or array.')
        if len(X) != len(C):
            raise ValueError('The length of X and C must match.')
        if len(X) != len(N):
            raise ValueError('The length of X, C and N must match.')

        C_upper = []
        for item in C:
            if type(item) in [str, np.str_]:
                C_upper.append(item.upper())  # for strings
            else:
                C_upper.append(item)  # for numbers
        C_unique = np.unique(C_upper)
        if len(C_unique) > 2:
            error_str = str('A maximum of 2 unique censoring codes are allowed. Within C there were ' + str(len(C_unique)) + ' unique values: ' + str(C_unique))
            raise ValueError(error_str)

        if type(failure_code) in [str, np.str_]:  # need to upper() the input since we are comparing with C_upper
            failure_code = failure_code.upper()
        if type(censor_code) in [str, np.str_]:
            censor_code = censor_code.upper()

        failures = np.array([])
        right_censored = np.array([])
        for i, c in enumerate(C_upper):
            if c == failure_code:
                failures = np.append(failures, np.ones(int(N[i])) * X[i])
            elif c == censor_code:
                right_censored = np.append(right_censored, np.ones(int(N[i])) * X[i])
            elif c in ['F', 'FAIL', 'FAILED', 0]:
                failures = np.append(failures, np.ones(int(N[i])) * X[i])
            elif c in ['R', 'RC', 'RIGHT CENS', 'RIGHT CENSORED', 'C', 'CENSORED', 'CENS', 'S', 'SUSP', 'SUSPENSION', 'SUSPENDED', 'UF', 'UNFAILED', 'UNFAIL', 'NF', 'NO FAIL', 'NO FAILURE', 'NOT FAILED', 1]:
                right_censored = np.append(right_censored, np.ones(int(N[i])) * X[i])
            else:
                raise ValueError('Unrecognised value in C. Common values are used as defaults but C contained an unrecognised values. You can fix this by specifying the arguments censor_code and failure_code.')
        if len(right_censored) == 0:
            right_censored = None
        self.failures = failures
        self.right_censored = right_censored
        # make the dataframe for printing and writing to excel
        if self.right_censored is not None:
            f, rc = list(self.failures), list(self.right_censored)
            len_f, len_rc = len(f), len(rc)
            max_len = max(len_f, len_rc)
            if not max_len == len_f:
                f.extend([''] * (max_len - len_f))
            if not max_len == len_rc:
                rc.extend([''] * (max_len - len_rc))
            Data = {'failures': f, 'right censored': rc}
            self.__df = pd.DataFrame(Data, columns=['failures', 'right censored'])
        else:
            Data = {'failures': self.failures}
            self.__df = pd.DataFrame(Data, columns=['failures'])

    def print(self):
        colorprint('Data (FR format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class FR_to_XCN:
    '''
    FR_to_XCN data format converter

    Inputs:
    failures - array or list
    right_censored -  array or list. Optional input.
    censor_code - the int or str to use for the censored items. Default is 'C'
    failure_code - the int or str to use for the failed items. Default is 'F'

    Output:
    X - event time
    C - censor code
    N - number of events at each event time

    Methods:
    print() - this will print a dataframe of the data in XCN format to the console
    write_to_xlsx() - this will export the data in XCN format to an xlsx file at the specified path.

    Example usage:
    XCN = FR_to_XCN(failures=[1,1,2,2,3], right_censored=[9,9,9,9,8,8,7])
    print(XCN.X)
        >>> [1 2 3 7 8 9]
    print(XCN.C)
        >>> ['F' 'F' 'F' 'C' 'C' 'C']
    print(XCN.N)
       >>> [2 2 1 1 2 4]
    XCN.print()
       >>> Data (XCN format)
           event time censor code  number of events
                    1           F                 2
                    2           F                 2
                    3           F                 1
                    7           C                 1
                    8           C                 2
                    9           C                 4

    '''

    def __init__(self, failures, right_censored=None, censor_code='C', failure_code='F'):
        if type(failures) not in [list, np.ndarray]:
            raise ValueError('failures must be a list or array.')
        if right_censored is not None:
            if type(right_censored) not in [list, np.ndarray]:
                raise ValueError('right_censored must be a list or array.')
            FNRN = FR_to_FNRN(failures=failures, right_censored=right_censored)
            self.X = np.hstack([FNRN.failures, FNRN.right_censored])
            self.N = np.hstack([FNRN.num_failures, FNRN.num_right_censored])
            if type(failure_code) not in [str, float, int, np.float64]:
                raise ValueError("failure_code must be a string or number. Default is 'F'")
            if type(censor_code) not in [str, float, int, np.float64]:
                raise ValueError("censor_code must be a string or number. Default is 'C'")
            F_cens = [failure_code] * len(FNRN.failures)
            C_cens = [censor_code] * len(FNRN.right_censored)
            self.C = np.hstack([F_cens, C_cens])
        else:
            FNRN = FR_to_FNRN(failures=failures)
            self.X = FNRN.failures
            self.N = FNRN.num_failures
            if type(failure_code) not in [str, float, int, np.float64]:
                raise ValueError("failure_code must be a string or number. Default is 'F'")
            self.C = np.array([failure_code] * len(FNRN.failures))
        Data = {'event time': self.X, 'censor code': self.C, 'number of events': self.N}
        self.__df = pd.DataFrame(data=Data, columns=['event time', 'censor code', 'number of events'])

    def print(self):
        colorprint('Data (XCN format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class FNRN_to_XCN:
    '''
    FNRN_to_XCN data format converter

    Inputs:
    failures - array or list
    num_failures - array or list. Length must match length of failures
    right_censored -  array or list. Optional input.
    num_right_censored - array or list. Optional Input. Length must match length of right_censored
    censor_code - the int or str to use for the censored items. Default is 'C'
    failure_code - the int or str to use for the failed items. Default is 'F'

    Output:
    X - event time
    C - censor code
    N - number of events at each event time

    Methods:
    print() - this will print a dataframe of the data in XCN format to the console
    write_to_xlsx() - this will export the data in XCN format to an xlsx file at the specified path.

    Example usage:
    XCN = FNRN_to_XCN(failures=[1, 2, 3], num_failures=[2, 2, 1], right_censored=[9, 8, 7], num_right_censored=[3, 2, 1])
    print(XCN.X)
        >>> [1. 2. 3. 7. 8. 9.]
    print(XCN.C)
        >>> ['F' 'F' 'F' 'C' 'C' 'C']
    print(XCN.N)
       >>> [2 2 1 1 2 3]
    XCN.print()
       >>> Data (XCN format)
           event time censor code  number of events
                    1           F                 2
                    2           F                 2
                    3           F                 1
                    7           C                 1
                    8           C                 2
                    9           C                 3

    '''

    def __init__(self, failures, num_failures, right_censored=None, num_right_censored=None, censor_code='C', failure_code='F'):
        if type(failures) not in [list, np.ndarray]:
            raise ValueError('failures must be a list or array.')
        if type(num_failures) not in [list, np.ndarray]:
            raise ValueError('num_failures must be a list or array.')
        if len(failures) != len(num_failures):
            raise ValueError('failures and num_failures must be the same length.')

        if right_censored is not None:
            if type(right_censored) not in [list, np.ndarray]:
                raise ValueError('right_censored must be a list or array.')
            if type(num_right_censored) not in [list, np.ndarray]:
                raise ValueError('num_right_censored must be a list or array.')
            if len(right_censored) != len(num_right_censored):
                raise ValueError('right_censored and num_right_censored must be the same length.')
            FR = FNRN_to_FR(failures=failures, num_failures=num_failures, right_censored=right_censored, num_right_censored=num_right_censored)
            FNRN = FR_to_FNRN(failures=FR.failures, right_censored=FR.right_censored)  # we do this seeming redundant conversion to combine any duplicates from FNRN which were not correctly summarized in the input data
            self.X = np.hstack([FNRN.failures, FNRN.right_censored])
            self.N = np.hstack([FNRN.num_failures, FNRN.num_right_censored])
            if type(failure_code) not in [str, float, int, np.float64]:
                raise ValueError("failure_code must be a string or number. Default is 'F'")
            if type(censor_code) not in [str, float, int, np.float64]:
                raise ValueError("censor_code must be a string or number. Default is 'C'")
            F_cens = [failure_code] * len(FNRN.failures)
            C_cens = [censor_code] * len(FNRN.right_censored)
            self.C = np.hstack([F_cens, C_cens])
        else:
            FR = FNRN_to_FR(failures=failures, num_failures=num_failures)
            FNRN = FR_to_FNRN(failures=FR.failures)  # we do this seeming redundant conversion to combine any duplicates from FNRN which were not correctly summarized in the input data
            self.X = FNRN.failures
            self.N = FNRN.num_failures
            if type(failure_code) not in [str, float, int, np.float64]:
                raise ValueError("failure_code must be a string or number. Default is 'F'")
            self.C = np.array([failure_code] * len(FNRN.failures))
        # make the dataframe for printing and writing to excel
        Data = {'event time': self.X, 'censor code': self.C, 'number of events': self.N}
        self.__df = pd.DataFrame(data=Data, columns=['event time', 'censor code', 'number of events'])

    def print(self):
        colorprint('Data (XCN format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class FR_to_FNRN:
    '''
    FR_to_FNRN data format converter

    Inputs:
    failures - array or list
    right censored -  array or list. Optional input.

    Output:
    failures
    num_failures
    right_censored
    num_right_censored

    Methods:
    print() - this will print a dataframe of the data in FNRN format to the console
    write_to_xlsx() - this will export the data in FNRN format to an xlsx file at the specified path.

    Example usage:
    FNRN = FR_to_FNRN(failures=[1,1,2,2,3], right_censored=[9,9,9,9,8,8,7])
    print(FNRN.failures)
        >>> [1 2 3]
    print(FNRN.num_failures)
        >>> [2 2 1]
    print(FNRN.right_censored)
       >>> [7 8 9]
    print(FNRN.num_right_censored)
       >>> [1 2 4]
    FNRN.print()
       >>> Data (FNRN format)
           failures  number of failures  right censored  number of right censored
                  1                   2               7                         1
                  2                   2               8                         2
                  3                   1               9                         4

    '''

    def __init__(self, failures, right_censored=None):
        if type(failures) not in [list, np.ndarray]:
            raise ValueError('failures must be a list or array.')
        self.failures, self.num_failures = np.unique(failures, return_counts=True)
        if right_censored is not None:
            if type(right_censored) not in [list, np.ndarray]:
                raise ValueError('right_censored must be a list or array.')
            self.right_censored, self.num_right_censored = np.unique(right_censored, return_counts=True)
        else:
            self.right_censored = None
            self.num_right_censored = None
        # make the dataframe for printing and writing to excel
        if self.right_censored is not None:
            f, nf, rc, nrc = list(self.failures), list(self.num_failures), list(self.right_censored), list(self.num_right_censored)
            len_f, len_rc = len(f), len(rc)
            max_len = max(len_f, len_rc)
            if not max_len == len_f:
                f.extend([''] * (max_len - len_f))
                nf.extend([''] * (max_len - len_f))
            if not max_len == len_rc:
                rc.extend([''] * (max_len - len_rc))
                nrc.extend([''] * (max_len - len_rc))
            Data = {'failures': f, 'number of failures': nf, 'right censored': rc, 'number of right censored': nrc}
            self.__df = pd.DataFrame(Data, columns=['failures', 'number of failures', 'right censored', 'number of right censored'])
        else:
            Data = {'failures': self.failures, 'number of failures': self.num_failures}
            self.__df = pd.DataFrame(Data, columns=['failures', 'number of failures'])

    def print(self):
        colorprint('Data (FNRN format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)


class FNRN_to_FR:
    '''
    FNRN_to_FR data format converter

    Inputs:
    failures - array or list
    num_failures - array or list. Length must match length of failures
    right_censored -  array or list. Optional input.
    num_right_censored - array or list. Optional Input. Length must match length of right_censored

    Output:
    failures
    right_censored

    Methods:
    print() - this will print a dataframe of the data in FR format to the console
    write_to_xlsx() - this will export the data in FR format to an xlsx file at the specified path.

    Example usage:
    FR = FNRN_to_FR(failures=[1,2,3], num_failures=[1,1,2], right_censored=[9,8,7], num_right_censored=[5,4,4])
    print(FR.failures)
        >>> [1. 2. 3. 3.]
    print(FR.right_censored)
       >>> [9. 9. 9. 9. 9. 8. 8. 8. 8. 7. 7. 7. 7.]
    FR.print()
       >>>  Data (FR format)
            failures  right censored
                   1               9
                   2               9
                   3               9
                   3               9
                                   9
                                   8
                                   8
                                   8
                                   8
                                   7
                                   7
                                   7
                                   7

    '''

    def __init__(self, failures, num_failures, right_censored=None, num_right_censored=None):
        if type(failures) not in [list, np.ndarray]:
            raise ValueError('failures must be a list or array.')
        if type(num_failures) not in [list, np.ndarray]:
            raise ValueError('num_failures must be a list or array.')
        if len(failures) != len(num_failures):
            raise ValueError('failures and num_failures must be the same length.')

        failures_out = np.array([])
        for i, f in enumerate(failures):
            failures_out = np.append(failures_out, np.ones(int(num_failures[i])) * f)
        self.failures = failures_out

        if right_censored is not None:
            if type(right_censored) not in [list, np.ndarray]:
                raise ValueError('right_censored must be a list or array.')
            if type(num_right_censored) not in [list, np.ndarray]:
                raise ValueError('num_right_censored must be a list or array.')
            if len(right_censored) != len(num_right_censored):
                raise ValueError('right_censored and num_right_censored must be the same length.')
            right_censored_out = np.array([])
            for i, rc in enumerate(right_censored):
                right_censored_out = np.append(right_censored_out, np.ones(int(num_right_censored[i])) * rc)
            self.right_censored = right_censored_out

            f, rc = list(self.failures), list(self.right_censored)
            len_f, len_rc = len(f), len(rc)
            max_len = max(len_f, len_rc)
            if not max_len == len_f:
                f.extend([''] * (max_len - len_f))
            if not max_len == len_rc:
                rc.extend([''] * (max_len - len_rc))
            Data = {'failures': f, 'right censored': rc}
            self.__df = pd.DataFrame(Data, columns=['failures', 'right censored'])
        else:
            self.right_censored = None
            Data = {'failures': self.failures}
            self.__df = pd.DataFrame(Data, columns=['failures'])

    def print(self):
        colorprint('Data (FR format)', bold=True, underline=True)
        print(self.__df.to_string(index=False), '\n')

    def write_to_xlsx(self, path, **kwargs):
        write_df_to_xlsx(df=self.__df, path=path, **kwargs)
