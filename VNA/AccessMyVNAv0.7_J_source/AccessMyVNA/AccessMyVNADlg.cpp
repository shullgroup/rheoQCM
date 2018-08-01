// AccessMyVNADlg.cpp : implementation file
//

#include "stdafx.h"
#include <complex>
#include "AccessMyVNA.h"
#include "..\AccessMyVNAdll\AccessMyVNAdll.h"
#include "AccessMyVNADlg.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm> 
#include <stdio.h>
#include <iomanip>    // Needed for stream modifiers fixed and setprecision
#include <iostream>
#include <time.h> 
#include <windows.h>
#include <tchar.h>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstdlib>



using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CAccessMyVNADlg dialog


CAccessMyVNADlg::CAccessMyVNADlg(CWnd* pParent /*=NULL*/)
	: CDialog(CAccessMyVNADlg::IDD, pParent)
	, nScanSteps(0)
	, nScanAverage(0)
	, dFreqStart(0)
	, dFreqStop(0)
	, strData(_T(""))
	, nParameter(0)
	, bAutoLoadResults(FALSE)
	, nInstrumentMode(0)
	, nOtherSelection(0)
	, sOther1(_T(""))
	, sOther2(_T(""))
	, sOther3(_T(""))
	, sOther4(_T(""))
	, sOther5(_T(""))
	, sOther6(_T(""))
	, sRetcode(_T(""))
	, sIndex(_T("0"))
	, nDisplayMode(0)
	, sIndexString(_T(""))
	, nStringFunction(0)
	, sStringParameter(_T(""))
	, sRetvalString(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CAccessMyVNADlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_SCANPOINTS, nScanSteps);
	DDX_Text(pDX, IDC_EDIT_SCANAVG, nScanAverage);
	DDV_MinMaxLong(pDX, nScanAverage, 0, 20);
	DDV_MinMaxLong(pDX, nScanSteps, 0, 50000);
	DDX_Text(pDX, IDC_EDIT_FSTART, dFreqStart);
	DDX_Text(pDX, IDC_EDIT_FSTOP, dFreqStop);
	DDX_Text(pDX, IDC_RICHEDIT21, strData);
	DDX_CBIndex(pDX, IDC_COMBO_PARAMETER_TYPE, nParameter);
	DDX_Check(pDX, IDC_CHECK1, bAutoLoadResults);
	DDX_Text(pDX, IDC_EDIT_INSTRUMENTMODE, nInstrumentMode);
	DDX_CBIndex(pDX, IDC_COMBO_OTHERS, nOtherSelection);
	DDX_Text(pDX, IDC_EDIT_OTHER1, sOther1);
	DDX_Text(pDX, IDC_EDIT_OTHER2, sOther2);
	DDX_Text(pDX, IDC_EDIT_OTHER3, sOther3);
	DDX_Text(pDX, IDC_EDIT_OTHER4, sOther4);
	DDX_Text(pDX, IDC_EDIT_OTHER5, sOther5);
	DDX_Text(pDX, IDC_EDIT_OTHER6, sOther6);
	DDX_Text(pDX, IDC_EDIT_INDEXVALUE, sIndex);
	DDX_Text(pDX, IDC_EDIT_RETVALUE, sRetcode);
	DDX_Text(pDX, IDC_EDIT_DISPLAYMODE, nDisplayMode);
	DDX_Text(pDX, IDC_EDIT_INDEXVALUE2, sIndexString);
	DDX_CBIndex(pDX, IDC_COMBO_STRING_OTHERS, nStringFunction);
	DDX_Text(pDX, IDC_EDIT_STRINGS, sStringParameter);
	DDX_Text(pDX, IDC_EDIT_RETVALUE2, sRetvalString);
}

BEGIN_MESSAGE_MAP(CAccessMyVNADlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BUTTON_INIT, &CAccessMyVNADlg::OnBnClickedButtonInit)
	ON_BN_CLICKED(IDC_BUTTON1, &CAccessMyVNADlg::OnBnClickedButtonSingle)
	ON_BN_CLICKED(IDC_BUTTON_SETSCANPOINTS, &CAccessMyVNADlg::OnBnClickedButtonSetScanSteps)
	ON_BN_CLICKED(IDC_BUTTON_GETSCANPOINTS, &CAccessMyVNADlg::OnBnClickedButtonGetScanSteps)
	ON_BN_CLICKED(IDC_BUTTON_SETSCANAVG, &CAccessMyVNADlg::OnBnClickedButtonSetscanavg)
	ON_BN_CLICKED(IDC_BUTTON_GETSCANAVG, &CAccessMyVNADlg::OnBnClickedButtonGetscanavg)
	ON_BN_CLICKED(IDC_BUTTON_SETFREQ, &CAccessMyVNADlg::OnBnClickedButtonSetfreq)
	ON_BN_CLICKED(IDC_BUTTON_GETFREQ, &CAccessMyVNADlg::OnBnClickedButtonGetfreq)
	ON_BN_CLICKED(IDC_BUTTON2, &CAccessMyVNADlg::OnBnClickedButtonShow)
	ON_BN_CLICKED(IDC_BUTTON3, &CAccessMyVNADlg::OnBnClickedButtonMinimize)
	ON_BN_CLICKED(IDC_BUTTON4, &CAccessMyVNADlg::OnBnClickedGetScanData)
	ON_COMMAND( MESSAGE_SCAN_ENDED, &CAccessMyVNADlg::ScanEnded)
	ON_CBN_SELCHANGE(IDC_COMBO_PARAMETER_TYPE, &CAccessMyVNADlg::OnCbnSelchangeComboParameterType)
	ON_BN_CLICKED(IDC_BUTTON_SETINSTRMODE, &CAccessMyVNADlg::OnBnClickedButtonSetinstrmode)
	ON_BN_CLICKED(IDC_BUTTON_GETINSTRMODE, &CAccessMyVNADlg::OnBnClickedButtonGetinstrmode)
	ON_BN_CLICKED(IDC_BUTTON_LOADCONFIG, &CAccessMyVNADlg::OnBnClickedButtonLoadconfig)
	ON_BN_CLICKED(IDC_BUTTON_SAVECONFIG, &CAccessMyVNADlg::OnBnClickedButtonSaveconfig)
	ON_BN_CLICKED(IDC_BUTTON_LOADCALIB, &CAccessMyVNADlg::OnBnClickedButtonLoadcalib)
	ON_BN_CLICKED(IDC_BUTTON_SAVECALIB, &CAccessMyVNADlg::OnBnClickedButtonSavecalib)
	ON_CBN_SELCHANGE(IDC_COMBO_OTHERS, &CAccessMyVNADlg::OnCbnSelchangeComboOthers)
	ON_BN_CLICKED(IDC_BUTTON_SETOTHERS, &CAccessMyVNADlg::OnBnClickedButtonSetothers)
	ON_BN_CLICKED(IDC_BUTTON_GETOTHERS, &CAccessMyVNADlg::OnBnClickedButtonGetothers)
	ON_BN_CLICKED(IDC_BUTTON_AUTOSCALE, &CAccessMyVNADlg::OnBnClickedButtonAutoscale)
	ON_BN_CLICKED(IDC_BUTTON_CLIPBOARD_COPY, &CAccessMyVNADlg::OnBnClickedButtonClipboardCopy)
	ON_BN_CLICKED(IDC_BUTTON_SAVESCAN, &CAccessMyVNADlg::OnBnClickedButtonSavescan)
	ON_BN_CLICKED(IDC_BUTTON_DISPLAYMODE, &CAccessMyVNADlg::OnBnClickedButtonSetDisplaymode)
	ON_BN_CLICKED(IDC_BUTTON_GETDISPLAYMODE, &CAccessMyVNADlg::OnBnClickedButtonGetdisplaymode)
	ON_BN_CLICKED(IDC_BUTTON_REFINE, &CAccessMyVNADlg::OnBnClickedButtonRefine)
	ON_BN_CLICKED(IDC_BUTTON_REFINE_CHOOSE_LOG, &CAccessMyVNADlg::OnBnClickedButtonRefineChooseLog)
	ON_BN_CLICKED(IDC_BUTTON_REFINE_LOG, &CAccessMyVNADlg::OnBnClickedButtonRefineLog)
	ON_BN_CLICKED(IDC_BUTTON_GET_EQ_CCT_DATA, &CAccessMyVNADlg::OnBnClickedButtonGetEqCctData)
	ON_BN_CLICKED(IDC_BUTTON_SETSTRING, &CAccessMyVNADlg::OnBnClickedButtonSetstring)
	ON_BN_CLICKED(IDC_BUTTON_GETSTRING, &CAccessMyVNADlg::OnBnClickedButtonGetstring)
	ON_EN_CHANGE(IDC_RICHEDIT21, &CAccessMyVNADlg::OnEnChangeRichedit21)
	ON_EN_CHANGE(IDC_EDIT_SCANPOINTS, &CAccessMyVNADlg::OnEnChangeEditScanpoints)
	ON_EN_CHANGE(IDC_EDIT_FSTART, &CAccessMyVNADlg::OnEnChangeEditFstart)
	ON_STN_CLICKED(IDC_STATIC_SCANSTATUS, &CAccessMyVNADlg::OnStnClickedStaticScanstatus)
	ON_BN_CLICKED(IDOK, &CAccessMyVNADlg::OnBnClickedOk)
END_MESSAGE_MAP()


// CAccessMyVNADlg message handlers

BOOL CAccessMyVNADlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	MyVNAShowWindow(1);//automatically intialize the myVNA program
	OnBnClickedButtonGetScanSteps();
	OnBnClickedButtonGetscanavg();
	clock_t goal = 3000 + clock();
	while (goal > clock());
	OnBnClickedButtonSingle();//automatically begin scanning

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CAccessMyVNADlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CAccessMyVNADlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CAccessMyVNADlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CAccessMyVNADlg::OnBnClickedButtonInit()
{
	if( MyVNAInit() == 0 )
	{
		MyVNAShowWindow(1);
		OnBnClickedButtonGetScanSteps();
		OnBnClickedButtonGetscanavg();
	//	OnBnClickedButtonGetfreq();
	//	OnBnClickedButtonGetinstrmode();
	//	OnBnClickedButtonGetdisplaymode();
	}
}

void CAccessMyVNADlg::OnBnClickedButtonSingle()
{
	int n;
	int state=1;//toggle state of scan
	float start1;//start frequency
	float end1;//end frequency
	int datapoints=200;//number of datapoints
	clock_t wait_time;//wait_time in between measurements
	clock_t wait_time2;//wait time in between sets of measurements, this is the time outside of the for loop
	int numharms;
	int open_state;
	int i;
	float currenth;
	float LB;//minimum frequency value for 1st harmonic
	float UB;//maximum frequency value for 1st harmonic
	n = 0;
	ifstream inFile;//create input file variable
	ifstream inrange;
	ifstream inopen_close;
	ifstream inwaittime2;
	//List of filenames
	std::string settings_txt = "\\settings.txt";//settings filename
	std::string cplusplusout_txt = "\\cplusplusout.txt";//cplusplusout filename
	std::string state1 = "\\state_matlab.txt";
	std::string WT_txt = "\\wait_time2.txt";//text file containing information for double wait_time2 value in text file is in microseconds
	//figure out relative path in c++
	char buff[200];//set max number of characters in the path
	_fullpath(buff, "", 200);//extract out current directory
	std::string rel_path1(buff);//convert current directory path from char variable to string variable
	std::string rel_path2(buff);//convert current directory path from char variable to string variable
	std::string rel_path3(buff);//convert current directory path from char variable to string variable
	std::string rel_path4(buff);//convert current directory path from char variable to string variable
	std::string fullpath_settings_txt = rel_path1.append(settings_txt);//append output file name onto current directory path
	std::string fullpath_cplusplusout_txt = rel_path2.append(cplusplusout_txt);
	std::string fullpath_state1 = rel_path3.append(state1);
	std::string fullpath_WT_txt = rel_path4.append(WT_txt);
	int nRet = MyVNASetScanSteps(datapoints);
	do{
		//Open the setting.txt and read in values
		inFile.open(fullpath_settings_txt);
		if (!inFile) {
			cout << "Unable to open file";
			exit(1); // terminate with error
		}
		//Read in the first value which is the toggle state of matlab "Start Scan" button
		inFile >> state;
		//Read in the amount of time to wait in between measurements
		inFile >> wait_time;
		//read in the total number of active harmonics
		inFile >> numharms;
		//Open the wait_time2.txt file to read in how much time to wait between each set of harmonics
		inwaittime2.open(fullpath_WT_txt);
		if (!inwaittime2) {
			cout << "Unable to open file";
			exit(1); // terminate with error
		}
		//Read in the wait time (in microseconds) in between each set of harmonics
		inwaittime2 >> wait_time2;		
		if (nRet)
			AfxMessageBox(IDS_CALL_ERROR);
		OnBnClickedButtonGetScanSteps();		
		for (i = 1; i <= numharms; i++){								
			std::string rel_path4(buff);//convert current directory path from char variable to string variable
			inFile >> currenth;
			if (currenth == 1){
				std::string settingharm = "\\settings01.txt";
				std::string fullpath_inrange = rel_path4.append(settingharm);
				inrange.open(fullpath_inrange);
				if (!inrange) {
					cout << "Unable to open file";
					exit(1); // terminate with error
				}
				inrange >> start1;
				inrange >> end1;
				inrange >> LB;
				inrange >> UB;
				inrange >> datapoints;
				inrange.close();
				int nRet = MyVNASetScanSteps(datapoints);
			}
			if (currenth == 3){
				std::string settingharm = "\\settings03.txt";
				std::string fullpath_inrange = rel_path4.append(settingharm);
				inrange.open(fullpath_inrange);
				if (!inrange) {
					cout << "Unable to open file";
					exit(1); // terminate with error
				}
				inrange >> start1;
				inrange >> end1;
				inrange >> LB;
				inrange >> UB;
				inrange >> datapoints;
				inrange.close();
				int nRet = MyVNASetScanSteps(datapoints);
			}
			if (currenth == 5){
				std::string settingharm = "\\settings05.txt";
				std::string fullpath_inrange = rel_path4.append(settingharm);
				inrange.open(fullpath_inrange);
				if (!inrange) {
					cout << "Unable to open file";
					exit(1); // terminate with error
				}
				inrange >> start1;
				inrange >> end1;
				inrange >> LB;
				inrange >> UB;
				inrange >> datapoints;
				inrange.close();
				int nRet = MyVNASetScanSteps(datapoints);
			}
			if (currenth == 7){
				std::string settingharm = "\\settings07.txt";
				std::string fullpath_inrange = rel_path4.append(settingharm);
				inrange.open(fullpath_inrange);
				if (!inrange) {
					cout << "Unable to open file";
					exit(1); // terminate with error
				}
				inrange >> start1;
				inrange >> end1;
				inrange >> LB;
				inrange >> UB;
				inrange >> datapoints;
				inrange.close();
				int nRet = MyVNASetScanSteps(datapoints);
			}
			if (currenth == 9){
				std::string settingharm = "\\settings09.txt";
				std::string fullpath_inrange = rel_path4.append(settingharm);
				inrange.open(fullpath_inrange);
				if (!inrange) {
					cout << "Unable to open file";
					exit(1); // terminate with error
				}
				inrange >> start1;
				inrange >> end1;
				inrange >> LB;
				inrange >> UB;
				inrange >> datapoints;
				inrange.close();
				int nRet = MyVNASetScanSteps(datapoints);
			}
			if (currenth == 11){
				std::string settingharm = "\\settings11.txt";
				std::string fullpath_inrange = rel_path4.append(settingharm);
				inrange.open(fullpath_inrange);
				if (!inrange) {
					cout << "Unable to open file";
					exit(1); // terminate with error
				}
				inrange >> start1;
				inrange >> end1;
				inrange >> LB;
				inrange >> UB;
				inrange >> datapoints;
				inrange.close();
				int nRet = MyVNASetScanSteps(datapoints);
			}

			//////SCAN AND OUTPUT DATA
			if (start1 > LB && start1 < UB && currenth == 1){
				double dG1[4000];
				double dB1[4000];
				//Conduct scan across set frequency
				int nRet1 = MyVNASetFequencies(start1*1e6, end1*1e6, 1);
				if (nRet1)
					AfxMessageBox(IDS_CALL_ERROR);
				((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T("Scanning..."));
				int nRet = MyVNASingleScan(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0);
				if (nRet)
					((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T(""));	
				clock_t goal0 = wait_time + clock();
				while (goal0 > clock());
				std::string freq_csv = "\\myVNAdata01.csv";//freq filename
				std::string state3 = "\\state_vna.txt";
				char buff[200];//set max number of characters in the path
				_fullpath(buff, "", 200);//extract out current directory
				std::string rel_path4(buff);//convert current directory path from char variable to string variable
				std::string rel_path5(buff);//convert current directory path from char variable to string variable
				std::string fullpath_freq_csv = rel_path4.append(freq_csv);//append output file name onto current directory path
				std::string fullpath_state = rel_path5.append(state3);//append output file name onto current directory path
				MyVNAGetScanData(0, datapoints-1, 15, 16, &dG1[0], &dB1[0]);//scan data
				MyVNAAutoscale();
				//clock_t goal = 1000 + clock();
				//while (goal > clock());
				ofstream f21(fullpath_freq_csv); // the output file for frequency
				copy(dG1, dG1 + datapoints, ostream_iterator<double>(f21, "\n"));
				copy(dB1, dB1 + datapoints, ostream_iterator<double>(f21, "\n"));
				//copy(dFreq, dFreq + 200, ostream_iterator<double>(f2, "\n"));
				f21.close();
			}
			if (start1 > LB && start1 < UB && currenth == 3){
				double dG3[4000];
				double dB3[4000];
				//Conduct scan across set frequency
				int nRet1 = MyVNASetFequencies(start1*1e6, end1*1e6, 1);
				if (nRet1)
					AfxMessageBox(IDS_CALL_ERROR);
				((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T("Scanning..."));
				int nRet = MyVNASingleScan(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0);
				if (nRet)
					((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T(""));
				clock_t goal0 = wait_time + clock();
				while (goal0 > clock());
				std::string freq_csv = "\\myVNAdata03.csv";//freq filename
				std::string state3 = "\\state_vna.txt";
				char buff[200];//set max number of characters in the path
				_fullpath(buff, "", 200);//extract out current directory
				std::string rel_path4(buff);//convert current directory path from char variable to string variable
				std::string rel_path5(buff);//convert current directory path from char variable to string variable
				std::string fullpath_freq_csv = rel_path4.append(freq_csv);//append output file name onto current directory path
				std::string fullpath_state = rel_path5.append(state3);//append output file name onto current directory path
				MyVNAGetScanData(0, datapoints - 1, 15, 16, &dG3[0], &dB3[0]);//scan data
				MyVNAAutoscale();
				//clock_t goal = 1000 + clock();
				//while (goal > clock());
				ofstream f23(fullpath_freq_csv); // the output file for frequency
				copy(dG3, dG3 + datapoints, ostream_iterator<double>(f23, "\n"));
				copy(dB3, dB3 + datapoints, ostream_iterator<double>(f23, "\n"));
				//copy(dFreq, dFreq + 200, ostream_iterator<double>(f2, "\n"));
				f23.close();
			}
			if (start1 > LB && start1 < UB && currenth == 5){
				double dG5[4000];
				double dB5[4000];
				//Conduct scan across set frequency
				int nRet1 = MyVNASetFequencies(start1*1e6, end1*1e6, 1);
				if (nRet1)
					AfxMessageBox(IDS_CALL_ERROR);
				((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T("Scanning..."));
				int nRet = MyVNASingleScan(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0);
				if (nRet)
					((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T(""));
				clock_t goal0 = wait_time + clock();
				while (goal0 > clock());
				std::string freq_csv = "\\myVNAdata05.csv";//freq filename
				std::string state3 = "\\state_vna.txt";
				char buff[200];//set max number of characters in the path
				_fullpath(buff, "", 200);//extract out current directory
				std::string rel_path4(buff);//convert current directory path from char variable to string variable
				std::string rel_path5(buff);//convert current directory path from char variable to string variable
				std::string fullpath_freq_csv = rel_path4.append(freq_csv);//append output file name onto current directory path
				std::string fullpath_state = rel_path5.append(state3);//append output file name onto current directory path
				MyVNAGetScanData(0, datapoints - 1, 15, 16, &dG5[0], &dB5[0]);//scan data
				MyVNAAutoscale();
				//clock_t goal = 1000 + clock();
				//while (goal > clock());
				ofstream f25(fullpath_freq_csv); // the output file for frequency
				copy(dG5, dG5 + datapoints, ostream_iterator<double>(f25, "\n"));
				copy(dB5, dB5 + datapoints, ostream_iterator<double>(f25, "\n"));
				//copy(dFreq, dFreq + 200, ostream_iterator<double>(f2, "\n"));
				f25.close();
			}
			if (start1 > LB && start1 < UB && currenth == 7){
				double dG7[4000];
				double dB7[4000];
				//Conduct scan across set frequency
				int nRet1 = MyVNASetFequencies(start1*1e6, end1*1e6, 1);
				if (nRet1)
					AfxMessageBox(IDS_CALL_ERROR);
				((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T("Scanning..."));
				int nRet = MyVNASingleScan(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0);
				if (nRet)
					((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T(""));
				clock_t goal0 = wait_time + clock();
				while (goal0 > clock());
				std::string freq_csv = "\\myVNAdata07.csv";//freq filename
				std::string state3 = "\\state_vna.txt";
				char buff[200];//set max number of characters in the path
				_fullpath(buff, "", 200);//extract out current directory
				std::string rel_path4(buff);//convert current directory path from char variable to string variable
				std::string rel_path5(buff);//convert current directory path from char variable to string variable
				std::string fullpath_freq_csv = rel_path4.append(freq_csv);//append output file name onto current directory path
				std::string fullpath_state = rel_path5.append(state3);//append output file name onto current directory path
				MyVNAGetScanData(0, datapoints - 1, 15, 16, &dG7[0], &dB7[0]);//scan data
				MyVNAAutoscale();
				//clock_t goal = 1000 + clock();
				//while (goal > clock());
				ofstream f27(fullpath_freq_csv); // the output file for frequency
				copy(dG7, dG7 + datapoints, ostream_iterator<double>(f27, "\n"));
				copy(dB7, dB7 + datapoints, ostream_iterator<double>(f27, "\n"));
				//copy(dFreq, dFreq + 200, ostream_iterator<double>(f2, "\n"));
				f27.close();
			}
			if (start1 > LB && start1 < UB && currenth == 9){
				double dG9[4000];
				double dB9[4000];
				//Conduct scan across set frequency
				int nRet1 = MyVNASetFequencies(start1*1e6, end1*1e6, 1);
				if (nRet1)
					AfxMessageBox(IDS_CALL_ERROR);
				((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T("Scanning..."));
				int nRet = MyVNASingleScan(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0);
				if (nRet)
					((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T(""));
				clock_t goal0 = wait_time + clock();
				while (goal0 > clock());
				std::string freq_csv = "\\myVNAdata09.csv";//freq filename
				std::string state3 = "\\state_vna.txt";
				char buff[200];//set max number of characters in the path
				_fullpath(buff, "", 200);//extract out current directory
				std::string rel_path4(buff);//convert current directory path from char variable to string variable
				std::string rel_path5(buff);//convert current directory path from char variable to string variable
				std::string fullpath_freq_csv = rel_path4.append(freq_csv);//append output file name onto current directory path
				std::string fullpath_state = rel_path5.append(state3);//append output file name onto current directory path
				MyVNAGetScanData(0, datapoints - 1, 15, 16, &dG9[0], &dB9[0]);//scan data
				MyVNAAutoscale();
				//clock_t goal = 1000 + clock();
				//while (goal > clock());
				ofstream f29(fullpath_freq_csv); // the output file for frequency
				copy(dG9, dG9 + datapoints, ostream_iterator<double>(f29, "\n"));
				copy(dB9, dB9 + datapoints, ostream_iterator<double>(f29, "\n"));
				//copy(dFreq, dFreq + 200, ostream_iterator<double>(f2, "\n"));
				f29.close();
			}
			if (start1 > LB && start1 < UB && currenth == 11){
				double dG11[4000];
				double dB11[4000];
				//Conduct scan across set frequency
				int nRet1 = MyVNASetFequencies(start1*1e6, end1*1e6, 1);
				if (nRet1)
					AfxMessageBox(IDS_CALL_ERROR);
				((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T("Scanning..."));
				int nRet = MyVNASingleScan(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0);
				if (nRet)
					((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T(""));
				clock_t goal0 = wait_time + clock();
				while (goal0 > clock());
				std::string freq_csv = "\\myVNAdata11.csv";//freq filename
				std::string state3 = "\\state_vna.txt";
				char buff[200];//set max number of characters in the path
				_fullpath(buff, "", 200);//extract out current directory
				std::string rel_path4(buff);//convert current directory path from char variable to string variable
				std::string rel_path5(buff);//convert current directory path from char variable to string variable
				std::string fullpath_freq_csv = rel_path4.append(freq_csv);//append output file name onto current directory path
				std::string fullpath_state = rel_path5.append(state3);//append output file name onto current directory path
				MyVNAGetScanData(0, datapoints - 1, 15, 16, &dG11[0], &dB11[0]);//scan data
				MyVNAAutoscale();
				//clock_t goal = 1000 + clock();
				//while (goal > clock());
				ofstream f211(fullpath_freq_csv); // the output file for frequency
				copy(dG11, dG11 + datapoints, ostream_iterator<double>(f211, "\n"));
				copy(dB11, dB11 + datapoints, ostream_iterator<double>(f211, "\n"));
				//copy(dFreq, dFreq + 200, ostream_iterator<double>(f2, "\n"));
				f211.close();
			}
			clock_t goal = wait_time + clock();
			while (goal > clock());
		}		
			clock_t goal = wait_time2 + clock();
			while (goal > clock());
			n++;
			inFile.close();//close the file
			inwaittime2.close();//close the file
			inopen_close.open(fullpath_state1);
			if (!inopen_close){
				cout << "Unabe to open state_malab.txt";
				exit(1);//terminate with error
			}
			//Read in the first value which determines whether or not AccessMyVNA should stay open or not
			inopen_close >> open_state;
			if (open_state == 0){
				OnOK();
			}
			inopen_close.close();//close the file		
			//end
	} while (state == 1);
	((CStatic *)GetDlgItem(IDC_STATIC_SCANSTATUS))->SetWindowText(_T("Not Scanning."));
}


void CAccessMyVNADlg::OnBnClickedButtonGetScanSteps()
{
	int nTemp;
	int nRet = MyVNAGetScanSteps(&nTemp);
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
	else
		nScanSteps = nTemp;
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonSetScanSteps()
{
	UpdateData( true );
	int nRet = MyVNASetScanSteps( nScanSteps );
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
}

void CAccessMyVNADlg::OnBnClickedButtonSetscanavg()
{
	UpdateData( true );
	int nRet = MyVNASetScanAverage( nScanAverage );
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
}

void CAccessMyVNADlg::OnBnClickedButtonGetscanavg()
{
	int nTemp;
	int nRet = MyVNAGetScanAverage( &nTemp);
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
	else
		nScanAverage = nTemp;
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonSetfreq()
{
	UpdateData( true );
	int nRet = MyVNASetFequencies( dFreqStart*1e6, dFreqStop*1e6, 1 );
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
}

void CAccessMyVNADlg::OnBnClickedButtonGetfreq()
{
	double dFreqData[10];
	if( MyVNAGetDoubleArray( GET_SCAN_FREQ_DATA, 0, 9, &dFreqData[0] ) == 0 )
	{
		dFreqStart = dFreqData[2] / 1e6;
		dFreqStop = dFreqData[3] / 1e6;
	}
	else
		AfxMessageBox( IDS_CALL_ERROR );
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonShow()
{
	MyVNAShowWindow(0);
}

void CAccessMyVNADlg::OnBnClickedButtonMinimize()
{
	MyVNAShowWindow(1);
}

void CAccessMyVNADlg::OnBnClickedGetScanData()
{
	UpdateData( true );
	strData = _T("");
	double dFreq[21];
	double dData[21];
	CString strTemp;
	if( MyVNAGetScanData( 0, 20, DISPLAY_FREQ_SCALE, nParameter, &dFreq[0], &dData[0] ) == 0 )
		for( int i=0; i<21; i++)
		{
			strTemp.Format(_T("%.6f    %.6f\r\n"), dFreq[i]/1e6, dData[i] );
			strData += strTemp;
		}
	UpdateData( false );
}

void  CAccessMyVNADlg::ScanEnded()
{
	((CStatic *)GetDlgItem( IDC_STATIC_SCANSTATUS ))->SetWindowText( _T("Scan Ended"));
}

void CAccessMyVNADlg::OnCbnSelchangeComboParameterType()
{
	UpdateData( true );
	if( bAutoLoadResults )
		OnBnClickedGetScanData();
}

void CAccessMyVNADlg::OnBnClickedButtonSetinstrmode()
{
	UpdateData( true );
	int nRet = MyVNASetInstrumentMode( nInstrumentMode );
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
}

void CAccessMyVNADlg::OnBnClickedButtonGetinstrmode()
{
	int nTemp;
	int nRet = MyVNAGetInstrumentMode( &nTemp);
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
	else
		nInstrumentMode = nTemp;
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonLoadconfig()
{
	CString fileName;
	INT_PTR nResult = IDOK;
	CFileDialog  dlgFile( TRUE, _T("txt"), NULL, OFN_EXPLORER | OFN_ENABLESIZING, _T("Configuration Files (*.txt)|*.txt|All Files (*.*)|*.*||") );
	dlgFile.GetOFN().lpstrFile = fileName.GetBuffer(4096);
	dlgFile.GetOFN().nMaxFile = 4096;
	nResult = dlgFile.DoModal();
	if( nResult != IDOK )
		return;
	MyVNALoadConfiguration( (_TCHAR *)(LPCTSTR)fileName );
}

void CAccessMyVNADlg::OnBnClickedButtonSaveconfig()
{
	CFileDialog  dlgFile( FALSE, _T("txt"), NULL, OFN_EXPLORER | OFN_ENABLESIZING, _T("Configuration Files (*.txt)|*.txt|All Files (*.*)|*.*||") );
	CString fileName;
	dlgFile.GetOFN().lpstrFile = fileName.GetBuffer(4096);
	dlgFile.GetOFN().nMaxFile = 4096;

	INT_PTR nResult = dlgFile.DoModal();
	if( nResult != IDOK )
		return;
	MyVNASaveConfiguration( (_TCHAR *)(LPCTSTR)fileName );
}

void CAccessMyVNADlg::OnBnClickedButtonLoadcalib()
{
	CString fileName;
	INT_PTR nResult = IDOK;
	CFileDialog  dlgFile( TRUE, _T("myVNA.cal"), NULL, OFN_EXPLORER | OFN_ENABLESIZING, _T("Calibration Files (*.myVNA.cal)|*.myVNA.cal|All Files (*.*)|*.*||") );
	dlgFile.GetOFN().lpstrFile = fileName.GetBuffer(4096);
	dlgFile.GetOFN().nMaxFile = 4096;
	nResult = dlgFile.DoModal();
	if( nResult != IDOK )
		return;
	MyVNALoadCalibration( (_TCHAR *)(LPCTSTR)fileName );
}

void CAccessMyVNADlg::OnBnClickedButtonSavecalib()
{
	CFileDialog  dlgFile( FALSE, _T("myVNA.cal"), NULL, OFN_EXPLORER | OFN_ENABLESIZING, _T("Calibration Files (*.myVNA.cal)|*.myVNA.cal|All Files (*.*)|*.*||") );
	CString fileName;
	dlgFile.GetOFN().lpstrFile = fileName.GetBuffer(4096);
	dlgFile.GetOFN().nMaxFile = 4096;

	INT_PTR nResult = dlgFile.DoModal();
	if( nResult != IDOK )
		return;
	MyVNASaveCalibration( (_TCHAR *)(LPCTSTR)fileName );
}

void CAccessMyVNADlg::OnCbnSelchangeComboOthers()
{
	UpdateData( true );
	sOther1 = sOther2 = sOther3 = sOther4 = sOther5 = sOther6 = sRetcode = _T("");
	bool bDouble = nOtherSelection >= 10 && nOtherSelection <= 12 ? true : false;
	((CStatic *)GetDlgItem( IDC_STATIC_FORMAT ))->SetWindowText( bDouble ? _T("Number format: Double") : _T("Number format: Hexadecimal"));
	sIndex = _T("0");
	UpdateData(false);
}

void CAccessMyVNADlg::OnBnClickedButtonSetothers()
{
	int nRetcode = 0;
	UpdateData( true );
	int nData[20];
	double dData[20];
	wchar_t *pcTemp;
	int nIndex = wcstoul( sIndex.GetBuffer(), &pcTemp, 16);
	nData[0] = wcstoul( sOther1.GetBuffer(), &pcTemp, 16);
	nData[1] = wcstoul( sOther2.GetBuffer(), &pcTemp, 16);
	nData[2] = wcstoul( sOther3.GetBuffer(), &pcTemp, 16);
	nData[3] = wcstoul( sOther4.GetBuffer(), &pcTemp, 16);
	nData[4] = wcstoul( sOther5.GetBuffer(), &pcTemp, 16);
	nData[5] = wcstoul( sOther6.GetBuffer(), &pcTemp, 16);
	dData[0] = wcstod( sOther1.GetBuffer(), &pcTemp );
	dData[1] = wcstod( sOther2.GetBuffer(), &pcTemp );
	dData[2] = wcstod( sOther3.GetBuffer(), &pcTemp );
	dData[3] = wcstod( sOther4.GetBuffer(), &pcTemp );
	dData[4] = wcstod( sOther5.GetBuffer(), &pcTemp );
	dData[5] = wcstod( sOther6.GetBuffer(), &pcTemp );
	switch( nOtherSelection )
	{
	case 0: // display options
		nRetcode =  MyVNASetIntegerArray( GETSET_DISPLAY_OPTIONS, nIndex, 4, &nData[0] );
		break;
	case 1: // print options
		nRetcode =  MyVNASetIntegerArray( GETSET_PRINT_OPTIONS, nIndex, 4, &nData[0] );
		break;
	case 2: // colour
		nRetcode =  MyVNASetIntegerArray( GETSET_COLOURS, nIndex, 1, &nData[0] );
		break;
	case 3: // left axis
		nRetcode =  MyVNASetIntegerArray( GETSETLEFTAXIS, nIndex, 4, &nData[0] );
		break;
	case 4: // right axis
		nRetcode =  MyVNASetIntegerArray( GETSETRIGHTAXIS, nIndex, 4, &nData[0] );
		break;
	case 5: // adc speed & delays
		nRetcode =  MyVNASetIntegerArray( GETSETHARDWARESPEEDS, nIndex, 4, &nData[0] );
		break;
	case 6: // hardware options
		nRetcode =  MyVNASetIntegerArray( GETSETHARDWAREOPTIONS, nIndex, 3, &nData[0] );
		break;
	case 7: // marker configuration
		nRetcode =  MyVNASetIntegerArray( GETSETMARKERSETUP, nIndex, 5, &nData[0] );
		break;
	case 8: // set constants not permitted
		nRetcode = -1;
		break;
	case 9: // equivalent circuit configuration
		nRetcode =  MyVNASetIntegerArray( GETSETEQCCTCONFIG, nIndex, 3, &nData[0] );
		break;
	case 10: // frequency display axis
		nRetcode =  MyVNASetDoubleArray( GETSET_DISPLAY_FT_AXIS_VALUES, nIndex, 4, &dData[0] );
		break;
	case 11: // left/right display axis
		nRetcode =  MyVNASetDoubleArray( GETSET_DISPLAY_VERTICAL_AXIS_VALUES, nIndex, 2, &dData[0] );
		break;
	case 12: // marker values
		nRetcode =  MyVNASetDoubleArray( GETSET_MARKER_VALUES, nIndex, 2, &dData[0] );
		break;
	case 13: // n2pk hardware values
		nRetcode =  MyVNASetDoubleArray( GETSET_N2PK_HARDWARE_VALUES, nIndex, 5, &dData[0] );
		break;
	case 14: // simulation setup
		nRetcode =  MyVNASetIntegerArray( GETSETSIMULATIONCONFIG, nIndex, 5, &nData[0] );
		break;
	case 15: // n2pk hardware values
		nRetcode =  MyVNASetDoubleArray( GETSET_SIMULATION_COMPONENT_VALUES, nIndex, 6, &dData[0] );
		break;
	case 16: // trace calculation options
		nRetcode =  MyVNASetIntegerArray( GETSETTRACECALCULATIONOPTIONS, nIndex, 1, &nData[0] );
		break;
	case 17: // get / set switch config
		nRetcode =  MyVNASetIntegerArray( GETSETSWITCHATTENUATORCONFIG, nIndex, 3, &nData[0] );
		break;
	case 18: // get / set switch settings
		nRetcode =  MyVNASetIntegerArray( GETSETSWITCHATTENUATORSETTINGS, nIndex, 2, &nData[0] );
		break;
	}
	sRetcode.Format( _T("%X"), nRetcode);
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonGetothers()
{
	int nRetcode = 0;
	UpdateData( true );
	int nData[20];
	double dData[20];
	wchar_t *pcTemp;
	int nParamCount = 0;
	bool bDouble = false;
	int nIndex = wcstoul( sIndex.GetBuffer(), &pcTemp, 16);
	switch( nOtherSelection )
	{
	case 0: // display options
		nRetcode =  MyVNAGetIntegerArray( GETSET_DISPLAY_OPTIONS, nIndex, nParamCount = 4, &nData[0] );
		break;
	case 1: // print options
		nRetcode =  MyVNAGetIntegerArray( GETSET_PRINT_OPTIONS, nIndex, nParamCount = 4, &nData[0] );
		break;
	case 2: // colour
		nRetcode =  MyVNAGetIntegerArray( GETSET_COLOURS, nIndex, nParamCount = 1, &nData[0] );
		break;
	case 3: // left axis
		nRetcode =  MyVNAGetIntegerArray( GETSETLEFTAXIS, nIndex, nParamCount = 4, &nData[0] );
		break;
	case 4: // right axis
		nRetcode =  MyVNAGetIntegerArray( GETSETRIGHTAXIS, nIndex, nParamCount = 4, &nData[0] );
		break;
	case 5:	// hardware speed / delays
		nRetcode =  MyVNAGetIntegerArray( GETSETHARDWARESPEEDS, nIndex, nParamCount = 4, &nData[0] );
		break;
	case 6: // hardware options
		nRetcode =  MyVNAGetIntegerArray( GETSETHARDWAREOPTIONS, nIndex, nParamCount = 3, &nData[0] );
		break;
	case 7: // hardware options
		nRetcode =  MyVNAGetIntegerArray( GETSETMARKERSETUP, nIndex, nParamCount = 6, &nData[0] );
		break;
	case 8: // hardware options
		nRetcode =  MyVNAGetIntegerArray( GETPROGRAMCONSTANTS, nIndex, nParamCount = 11, &nData[0] );
		break;
	case 9: // equivalent circuit configuration
		nRetcode =  MyVNAGetIntegerArray( GETSETEQCCTCONFIG, nIndex, nParamCount = 3, &nData[0] );
		break;
	case 10: // frequency display axis
		bDouble = true;
		nRetcode =  MyVNAGetDoubleArray( GETSET_DISPLAY_FT_AXIS_VALUES, nIndex, nParamCount = 4, &dData[0] );
		break;
	case 11: // left/right display axis
		bDouble = true;
		nRetcode =  MyVNAGetDoubleArray( GETSET_DISPLAY_VERTICAL_AXIS_VALUES, nIndex, nParamCount = 2, &dData[0] );
		break;
	case 12: // marker values
		bDouble = true;
		nRetcode =  MyVNAGetDoubleArray( GETSET_MARKER_VALUES, nIndex, nParamCount = 2, &dData[0] );
		break;
	case 13: // n2pk hrdware values
		bDouble = true;
		nRetcode =  MyVNAGetDoubleArray( GETSET_N2PK_HARDWARE_VALUES, nIndex, nParamCount = 5, &dData[0] );
		break;
	case 14: // simulation configuration
		nRetcode =  MyVNAGetIntegerArray( GETSETSIMULATIONCONFIG, nIndex, nParamCount = 5, &nData[0] );
		break;
	case 15: // simulation component values
		bDouble = true;
		nRetcode =  MyVNAGetDoubleArray( GETSET_SIMULATION_COMPONENT_VALUES, nIndex, nParamCount = 16, &dData[0] );
		break;
	case 16: // trace calculation options
		nRetcode =  MyVNAGetIntegerArray( GETSETTRACECALCULATIONOPTIONS, nIndex, nParamCount = 1, &nData[0] );
		break;
	case 17: // get switch & atten config
		nRetcode =  MyVNAGetIntegerArray( GETSETSWITCHATTENUATORCONFIG, nIndex, nParamCount = 3, &nData[0] );
		break;
	case 18: // get switch & atten settings
		nRetcode =  MyVNAGetIntegerArray( GETSETSWITCHATTENUATORSETTINGS, nIndex, nParamCount = 2, &nData[0] );
		break;
	}
	if( nRetcode == 0 )
	{
		if( bDouble )
		{
			sOther1.Format( nParamCount > 0 ? _T("%g") : _T(""), dData[0]);
			sOther2.Format( nParamCount > 1 ? _T("%g") : _T(""), dData[1]);
			sOther3.Format( nParamCount > 2 ? _T("%g") : _T(""), dData[2]);
			sOther4.Format( nParamCount > 3 ? _T("%g") : _T(""), dData[3]);
			sOther5.Format( nParamCount > 4 ? _T("%g") : _T(""), dData[4]);
			sOther6.Format( nParamCount > 5 ? _T("%g") : _T(""), dData[5]);
		}
		else
		{
			sOther1.Format( nParamCount > 0 ? _T("%X") : _T(""), nData[0]);
			sOther2.Format( nParamCount > 1 ? _T("%X") : _T(""), nData[1]);
			sOther3.Format( nParamCount > 2 ? _T("%X") : _T(""), nData[2]);
			sOther4.Format( nParamCount > 3 ? _T("%X") : _T(""), nData[3]);
			sOther5.Format( nParamCount > 4 ? _T("%X") : _T(""), nData[4]);
			sOther6.Format( nParamCount > 5 ? _T("%X") : _T(""), nData[5]);
		}
	}
	sRetcode.Format( _T("%X"), nRetcode);
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonAutoscale()
{
	MyVNAAutoscale();
}

void CAccessMyVNADlg::OnBnClickedButtonClipboardCopy()
{
	MyVNAClipboardCopy();
}

void CAccessMyVNADlg::OnBnClickedButtonSavescan()
{
	double dFreq[200];
	double dData[200];
	double dData2[200];
	double dummy[200];
	//List of filenames
	std::string freq_csv = "\\myVNAdata.csv";//freq filename
	//std::string myVNAdata_csv = "\\myVNAdata.csv";//1st dataset filename
	//std::string myVNAdata2_csv = "\\myVNAdata2.csv";//2nd dataset filename
	//figure out relative path in c++
	char buff[200];//set max number of characters in the path
	_fullpath(buff, "", 200);//extract out current directory
	std::string rel_path1(buff);//convert current directory path from char variable to string variable
	//std::string rel_path2(buff);//convert current directory path from char variable to string variable
	//std::string rel_path3(buff);//convert current directory path from char variable to string variable
	std::string fullpath_freq_csv = rel_path1.append(freq_csv);//append output file name onto current directory path
	//std::string fullpath_myVNAdata_csv = rel_path2.append(myVNAdata_csv);//append output file name onto current directory path
	//std::string fullpath_myVNAdata2_csv = rel_path3.append(myVNAdata2_csv);//append output file name onto current directory path
	//Now scan the data and output the data into a .csv file
	MyVNAGetScanData(0, 199, -1, 15, &dFreq[0], &dData[0]);//scan data
	MyVNAGetScanData(0, 199, DISPLAY_FREQ_SCALE, 16, &dummy[0], &dData2[0]);//scan data
	MyVNAAutoscale();
	ofstream f1(fullpath_freq_csv); // the output file for frequency
//	ofstream f1(fullpath_myVNAdata_csv); // the output file for first dataset
	copy(dData, dData + 200, ostream_iterator<double>(f1, "\n"));
//	ofstream f3(fullpath_myVNAdata2_csv); // the output file for first dataset
	copy(dData2, dData2 + 200, ostream_iterator<double>(f1, "\n"));
	copy(dFreq, dFreq + 200, ostream_iterator<double>(f1, "\n"));
	f1.close();
//	f2.close();
//	f3.close();
}

void CAccessMyVNADlg::OnBnClickedButtonSetDisplaymode()
{
	UpdateData( true );
	int nRet = MyVNASetDisplayMode( nDisplayMode );
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
}

void CAccessMyVNADlg::OnBnClickedButtonGetdisplaymode()
{
	int nTemp;
	int nRet = MyVNAGetDisplayMode( &nTemp);
	if( nRet )
		AfxMessageBox( IDS_CALL_ERROR );
	else
		nDisplayMode = nTemp;
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonRefine()
{
	((CStatic *)GetDlgItem( IDC_STATIC_SCANSTATUS ))->SetWindowText( _T("Refining..."));
	int nRet = MyVNAEqCctRefine(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0 );
	if( nRet != 0 )
		((CStatic *)GetDlgItem( IDC_STATIC_SCANSTATUS ))->SetWindowText( _T("Cancelled..."));
}

void CAccessMyVNADlg::OnBnClickedButtonRefineChooseLog()
{
	CFileDialog  dlgFile( FALSE, _T("csv"), NULL, OFN_EXPLORER | OFN_ENABLESIZING | OFN_DONTADDTORECENT, _T("CSV Data Set (*.csv)|*.csv|All Files (*.*)|*.*||") );
	CString fileName;
	dlgFile.GetOFN().lpstrFile = fileName.GetBuffer(4096);
	dlgFile.GetOFN().nMaxFile = 4096;

	INT_PTR nResult = dlgFile.DoModal();
	if( nResult == IDOK )
		MyVNASetEqCctLogFile( (_TCHAR *)(LPCTSTR)fileName );
}

void CAccessMyVNADlg::OnBnClickedButtonRefineLog()
{
	MyVNALogEqCctResults(_T("Dummy Description"));
}

void CAccessMyVNADlg::OnBnClickedButtonGetEqCctData()
{
	UpdateData( true );
	strData = _T("");
	double dData[32];
	CString strTemp;
	if( MyVNAGetDoubleArray( GET_EQCCT_VALUES, 0, 32, &dData[0] ) == 0 )
		for( int i=0; i<21; i++)
		{
			strTemp.Format(_T("%.9g\r\n"), dData[i] );
			strData += strTemp;
		}
	UpdateData( false );
}

void CAccessMyVNADlg::OnOK()
{
	MyVNAClose();

	CDialog::OnOK();
}

void CAccessMyVNADlg::OnCancel()
{
	MyVNAClose();

	CDialog::OnCancel();
}

void CAccessMyVNADlg::OnBnClickedButtonSetstring()
{
	int nRetcode = 0;
	UpdateData( true );
	wchar_t *pcTemp;
	int nIndex = wcstoul( sIndexString.GetBuffer(), &pcTemp, 16);
	switch( nStringFunction )
	{
	case 0: // equation
		nRetcode =  MyVNASetString( GETSET_EQUATION, nIndex, sStringParameter.GetBuffer() );
		break;
	}
	sRetvalString.Format( _T("%X"), nRetcode);
	UpdateData( false );
}

void CAccessMyVNADlg::OnBnClickedButtonGetstring()
{
	int nRetcode = 0;
	UpdateData( true );
	wchar_t *pcTemp;
	int nIndex = wcstoul( sIndexString.GetBuffer(), &pcTemp, 16);
	switch( nStringFunction )
	{
	case 0: // equation
		sStringParameter =  MyVNAGetString( GETSET_EQUATION, nIndex );
		break;
	}
	sRetvalString.Format( _T(""));
	UpdateData( false );
}


void CAccessMyVNADlg::OnEnChangeRichedit21()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}


void CAccessMyVNADlg::OnEnChangeEditScanpoints()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}


void CAccessMyVNADlg::OnEnChangeEditFstart()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}


void CAccessMyVNADlg::OnStnClickedStaticScanstatus()
{
	// TODO: Add your control notification handler code here
}




#define BUFSIZE 4096
#define LONG_DIR_NAME TEXT("c:\\longdirectoryname")

void _tmain(int argc, TCHAR *argv[])
{
	DWORD  retval = 0;
	BOOL   success;
	TCHAR  buffer[BUFSIZE] = TEXT("");
	TCHAR  buf[BUFSIZE] = TEXT("");
	TCHAR** lppPart = { NULL };

	if (argc != 2)
	{
		_tprintf(TEXT("Usage: %s [file]\n"), argv[0]);
		return;
	}

	// Retrieve the full path name for a file. 
	// The file does not need to exist.

	retval = GetFullPathName(argv[1],
		BUFSIZE,
		buffer,
		lppPart);

	if (retval == 0)
	{
		// Handle an error condition.
		printf("GetFullPathName failed (%d)\n", GetLastError());
		return;
	}
	else
	{
		_tprintf(TEXT("The full path name is:  %s\n"), buffer);
		if (lppPart != NULL && *lppPart != 0)
		{
			_tprintf(TEXT("The final component in the path name is:  %s\n"), *lppPart);
		}
	}
}


void CAccessMyVNADlg::OnBnClickedOk()
{
	// TODO: Add your control notification handler code here
	CDialog::OnOK();
	MyVNAClose();
}
