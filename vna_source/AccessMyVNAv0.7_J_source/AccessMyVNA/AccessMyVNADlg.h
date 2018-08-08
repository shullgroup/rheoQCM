// AccessMyVNADlg.h : header file
//

#pragma once


// CAccessMyVNADlg dialog
class CAccessMyVNADlg : public CDialog
{
// Construction
public:
	CAccessMyVNADlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_ACCESSMYVNA_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonInit();
	afx_msg void OnBnClickedButtonSingle();
	long nScanSteps;
	afx_msg void OnBnClickedButtonSetScanSteps();
	afx_msg void OnBnClickedButtonGetScanSteps();
	long nScanAverage;
	afx_msg void OnBnClickedButtonSetscanavg();
	afx_msg void OnBnClickedButtonGetscanavg();
	double dFreqStart;
	double dFreqStop;
	afx_msg void OnBnClickedButtonSetfreq();
	afx_msg void OnBnClickedButtonGetfreq();
	afx_msg void OnBnClickedButtonShow();
	afx_msg void OnBnClickedButtonMinimize();
	afx_msg void OnBnClickedGetScanData();
	CString strData;
	void ScanEnded();
	int nParameter;
	BOOL bAutoLoadResults;
	afx_msg void OnCbnSelchangeComboParameterType();
	afx_msg void OnBnClickedButtonSetinstrmode();
	afx_msg void OnBnClickedButtonGetinstrmode();
	int nInstrumentMode;
	afx_msg void OnBnClickedButtonLoadconfig();
	afx_msg void OnBnClickedButtonSaveconfig();
	afx_msg void OnBnClickedButtonLoadcalib();
	afx_msg void OnBnClickedButtonSavecalib();
	int nOtherSelection;
	CString sOther1;
	CString sOther2;
	CString sOther3;
	CString sOther4;
	CString sOther5;
	CString sOther6;
	CString sRetcode;
	CString sIndex;
	afx_msg void OnCbnSelchangeComboOthers();
	afx_msg void OnBnClickedButtonSetothers();
	afx_msg void OnBnClickedButtonGetothers();
	afx_msg void OnBnClickedButtonAutoscale();
	afx_msg void OnBnClickedButtonClipboardCopy();
	afx_msg void OnBnClickedButtonSavescan();
	int nDisplayMode;
	afx_msg void OnBnClickedButtonSetDisplaymode();
	afx_msg void OnBnClickedButtonGetdisplaymode();
	afx_msg void OnBnClickedButtonRefine();
	afx_msg void OnBnClickedButtonRefineChooseLog();
	afx_msg void OnBnClickedButtonRefineLog();
	afx_msg void OnBnClickedButtonGetEqCctData();
protected:
	virtual void OnOK();
	virtual void OnCancel();
public:
	CString sIndexString;
	afx_msg void OnBnClickedButtonSetstring();
	afx_msg void OnBnClickedButtonGetstring();
	int nStringFunction;
	CString sStringParameter;
	CString sRetvalString;
	afx_msg void OnEnChangeRichedit21();
	afx_msg void OnEnChangeEditScanpoints();
	afx_msg void OnEnChangeEditFstart();
	afx_msg void OnStnClickedStaticScanstatus();
	afx_msg void OnBnClickedOk();
};

#define MESSAGE_SCAN_ENDED (WM_USER+0x1234)