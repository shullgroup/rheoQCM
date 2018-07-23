// AccessMyVNAdll.cpp : Defines the initialization routines for the DLL.
//

#include "stdafx.h"
#include <complex>
#include "AccessMyVNAdll.h"
#include "process.h"
#include "comutil.h"
#define _WIN32_DCOM 

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//
//TODO: If this DLL is dynamically linked against the MFC DLLs,
//		any functions exported from this DLL which call into
//		MFC must have the AFX_MANAGE_STATE macro added at the
//		very beginning of the function.
//
//		For example:
//
//		extern "C" BOOL PASCAL EXPORT ExportedFunction()
//		{
//			AFX_MANAGE_STATE(AfxGetStaticModuleState());
//			// normal function body here
//		}
//
//		It is very important that this macro appear in each
//		function, prior to any calls into MFC.  This means that
//		it must appear as the first statement within the 
//		function, even before any object variable declarations
//		as their constructors may generate calls into the MFC
//		DLL.
//
//		Please see MFC Technical Notes 33 and 58 for additional
//		details.
//

// **************************************************************
// This is the identity we seek for myVNA

static const CLSID clsid_myVNA = 
{ 0x4d9364d3, 0x3187, 0x4cea, { 0xa2, 0x8d, 0x5d, 0xc0, 0x6e, 0x34, 0x33, 0x44 } };

// CAccessMyVNAdllApp

BEGIN_MESSAGE_MAP(CAccessMyVNAdllApp, CWinApp)
END_MESSAGE_MAP()


// CAccessMyVNAdllApp construction

CAccessMyVNAdllApp::CAccessMyVNAdllApp()
{
	m_autoMyVNAObject = NULL;
}

CAccessMyVNAdllApp::~CAccessMyVNAdllApp()
{
}


// The one and only CAccessMyVNAdllApp object

CAccessMyVNAdllApp theApp;


// CAccessMyVNAdllApp initialization & closedown

BOOL CAccessMyVNAdllApp::InitInstance()
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	CWinApp::InitInstance();
	m_autoMyVNAObject = new CAutoMyVNA;
	CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);

	return TRUE;
}

int CAccessMyVNAdllApp::ExitInstance()
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	// the following should be unnecessary as a prior call to MyVNAClose()
	// will have released the OLE server but we'll do it here for good measure
	// just in case. Note that if the call to close has not been made correctly
	// then we may hang at this point
	m_autoMyVNAObject->ReleaseDispatch();
	CoUninitialize();
	delete m_autoMyVNAObject;
	m_autoMyVNAObject = NULL;

	return CWinApp::ExitInstance();
}

// helper function used in getting data
int MyVNAGetHelper( _TCHAR * pString, enum VARENUM type, void *pnResult )
{

	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			{
				theApp.m_autoMyVNAObject->GetProperty(dispid, type, pnResult);
				errcode = 0;
			}
	}
	catch ( ... )
	{
		errcode = -1;
	}
	return errcode;
}

// helper function used in setting integer data
int MyVNASetHelperInt( _TCHAR * pString, enum VARENUM type, int Value )
{

	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			{
				theApp.m_autoMyVNAObject->SetProperty(dispid, type, Value);
				errcode = 0;
			}
	}
	catch ( ... )
	{
		errcode = -1;
	}
	return errcode;
}

// helper function used in setting double data
int MyVNASetHelperDouble( _TCHAR * pString, enum VARENUM type, double Value )
{

	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			{
				theApp.m_autoMyVNAObject->SetProperty(dispid, type, Value);
				errcode = 0;
			}
	}
	catch ( ... )
	{
		errcode = -1;
	}
	return errcode;
}

// execute a function (no parameters)
int MyVNACallFunctionVoid( _TCHAR * pString )
{

	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			{
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_EMPTY, NULL, NULL);
				errcode = 0;
			}
	}
	catch ( ... )
	{
		errcode = -1;
	}
	return errcode;
}

// execute a function (1 string parameter)
int MyVNACallFunctionBstr( _TCHAR * pStrFunction, _TCHAR * pStrParameter )
{
	static BYTE BASED_CODE parms[] = VTS_BSTR;
	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pStrFunction);
	BSTR BStrParameter = pStrParameter;

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			{
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_EMPTY, NULL, parms, BStrParameter );
				errcode = 0;
			}
	}
	catch ( ... )
	{
		errcode = -1;
	}
	return errcode;
}

// helper function to get an integer array of data
int MyVNAGetIntArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, int *pnResult )
{

	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	SAFEARRAYBOUND rgsabound;
	rgsabound.cElements = nArraySize;
	rgsabound.lLbound = 0;

	VARIANT nData;
	nData.vt = VT_ARRAY | VT_I4;

	SAFEARRAY FAR *pArray;

	pArray = SafeArrayCreate(VT_I4, 1, &rgsabound);

	nData.parray = pArray;

	for( int i=0; i< nArraySize; i++ )
		((int *)pArray->pvData)[i] = pnResult[i];

	static BYTE parms[] = VTS_I4 VTS_I4 VTS_I4 VTS_PVARIANT;

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, nWhat, nIndex, nArraySize, &nData );
	}
	catch ( ... )
	{
		errcode = -1;
	}
	if( errcode == 0 )
		for( int i=0; i< nArraySize; i++ )
			pnResult[i] = ((int *)nData.parray->pvData)[i];
	if (pArray != NULL )
		SafeArrayDestroy( pArray );
	return errcode;
}

// helper function to set an integer array of data
int MyVNASetIntArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, int *pnData )
{

	int errcode = 0;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	SAFEARRAYBOUND rgsabound;
	rgsabound.cElements = nArraySize;
	rgsabound.lLbound = 0;

	VARIANT nData;
	nData.vt = VT_ARRAY | VT_I4;

	SAFEARRAY FAR *pArray;

	pArray = SafeArrayCreate(VT_I4, 1, &rgsabound);
	if( pArray == NULL )
		return errcode;

	nData.parray = pArray;

	for( int i=0; i< nArraySize; i++ )
		((int *)pArray->pvData)[i] = pnData[i];

	static BYTE parms[] = VTS_I4 VTS_I4 VTS_I4 VTS_PVARIANT;

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, nWhat, nIndex, nArraySize, &nData );
	}
	catch ( ... )
	{
		errcode = -1;
	}
	if (pArray != NULL )
		SafeArrayDestroy( pArray );
	return errcode;
}

// helper function to get an array of double data
int MyVNAGetDoubleArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, double *pnResult )
{

	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	SAFEARRAYBOUND rgsabound;
	rgsabound.cElements = nArraySize;
	rgsabound.lLbound = 0;

	VARIANT nData;
	nData.vt = VT_ARRAY | VT_R8;

	SAFEARRAY FAR *pArray;

	pArray = SafeArrayCreate(VT_R8, 1, &rgsabound);

	nData.parray = pArray;

	static BYTE parms[] = VTS_I4 VTS_I4 VTS_I4 VTS_PVARIANT;

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, nWhat, nIndex, nArraySize, &nData );
	}
	catch ( ... )
	{
		errcode = -1;
	}
	if( errcode == 0 )
		for( int i=0; i< nArraySize; i++ )
			pnResult[i] = ((double *)pArray->pvData)[i];
	if (pArray != NULL )
		SafeArrayDestroy( pArray );
	return errcode;
}

// helper function to set an array of double data
int MyVNASetDoubleArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, double *pnData )
{

	int errcode = 0;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	SAFEARRAYBOUND rgsabound;
	rgsabound.cElements = nArraySize;
	rgsabound.lLbound = 0;

	VARIANT nData;
	nData.vt = VT_ARRAY | VT_R8;

	SAFEARRAY FAR *pArray;

	pArray = SafeArrayCreate(VT_R8, 1, &rgsabound);
	if( pArray == NULL )
		return errcode;

	nData.parray = pArray;

	for( int i=0; i< nArraySize; i++ )
		((double *)pArray->pvData)[i] = pnData[i];

	static BYTE parms[] = VTS_I4 VTS_I4 VTS_I4 VTS_PVARIANT;

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, nWhat, nIndex, nArraySize, &nData );
	}
	catch ( ... )
	{
		errcode = -1;
	}
	if (pArray != NULL )
		SafeArrayDestroy( pArray );
	return errcode;
}

// helper function to set an array of double data
int MyVNASetStringHelper( _TCHAR * pString, int nWhat, int nIndex, _TCHAR * sWhat )
{

	static BYTE BASED_CODE parms[] = VTS_I4 VTS_I4 VTS_BSTR;
	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);
	BSTR BStrParameter = sWhat;

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, nWhat, nIndex, BStrParameter );
	}
	catch ( ... )
	{
		errcode = -1;
	}
	return errcode;
}

// return a string

CString MyVNAGetStringHelper( _TCHAR * pString, int nWhat, int nIndex )
{
	static BYTE BASED_CODE parms[] = VTS_I4 VTS_I4;
	CString StringResult;
	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)pString);

	try
	{
		if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
			if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
				theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_BSTR, (void*)&StringResult, parms, nWhat, nIndex );
	}
	catch ( ... )
	{
		StringResult = _T("Error");
	}
	return StringResult;
}



// /////////////////////////////////////////////////////////////////////////////////////////////////
// The routines from here on are the exported interface functions

// Call this to initialize. It invokes a fresh copy of myVNA ready for use

__declspec(dllexport) int _stdcall MyVNAInit(void)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());

	if( theApp.m_autoMyVNAObject->m_lpDispatch == NULL )
	{
		COleException *e = new COleException;
		if (!theApp.m_autoMyVNAObject->CreateDispatch( clsid_myVNA, e ))
		{
#ifdef _DEBUG
			TRACE("%s(%d): OLE Execption caught: SCODE = %x", __FILE__, __LINE__, COleException::Process(e));
			AfxMessageBox(_T("Unable to create the 'myVNA.Document' object.  Please run myVNA to configure registry and try again."));
#endif
			e->Delete();
			return -1;  // fail
		}
		e->Delete();
	}
	return 0;
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// you MUST call this before the calling windows application closes in order to correctly shut down OLE

__declspec(dllexport) int _stdcall MyVNAClose(void)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	if( theApp.m_autoMyVNAObject != NULL )
		theApp.m_autoMyVNAObject->ReleaseDispatch();
	return 0;
}


// /////////////////////////////////////////////////////////////////////////////////////////////////
// scan related functions

// perform (or stop if in progress) a single scan

__declspec(dllexport) int _stdcall MyVNASingleScan(int Message, HWND hWnd, int nCommand, int lParam )
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	static BYTE BASED_CODE parms[] = VTS_I4 VTS_I4 VTS_I4 VTS_I4;
	int errcode = 0;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)_T("SingleScan"));
	if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
		if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, Message, hWnd, nCommand, lParam);
	return errcode;
}

// perform (or stop if in progress) a scan sequence for equivalent circuit refine

__declspec(dllexport) int _stdcall MyVNAEqCctRefine(int Message, HWND hWnd, int nCommand, int lParam )
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	static BYTE BASED_CODE parms[] = VTS_I4 VTS_I4 VTS_I4 VTS_I4;
	int errcode = 0;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)_T("Eq Cct Refine"));
	if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
		if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, Message, hWnd, nCommand, lParam);
	return errcode;
}


// /////////////////////////////////////////////////////////////////////////////////////////////////
// get and set the number of scan steps

__declspec(dllexport) int _stdcall MyVNASetScanSteps(int nSteps)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNASetHelperInt( _T("ScanPoints"), VT_I4, nSteps );
}

__declspec(dllexport) int _stdcall MyVNAGetScanSteps(int *pnSteps)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNAGetHelper( _T("ScanPoints"), VT_I4, pnSteps );
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// get and set the instrument mode

__declspec(dllexport) int _stdcall MyVNASetInstrumentMode(int nMode)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNASetHelperInt( _T("InstrumentMode"), VT_I4, nMode );
}

__declspec(dllexport) int _stdcall MyVNAGetInstrumentMode(int *pnMode)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNAGetHelper( _T("InstrumentMode"), VT_I4, pnMode );
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// get and set the display mode

__declspec(dllexport) int _stdcall MyVNASetDisplayMode(int nMode)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNASetHelperInt( _T("DisplayMode"), VT_I4, nMode );
}

__declspec(dllexport) int _stdcall MyVNAGetDisplayMode(int *pnMode)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNAGetHelper( _T("DisplayMode"), VT_I4, pnMode );
}


// /////////////////////////////////////////////////////////////////////////////////////////////////
// get and set averaging counter

__declspec(dllexport) int _stdcall MyVNASetScanAverage(int nAverage)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNASetHelperInt( _T("ScanAverage"), VT_I4, nAverage );
}

__declspec(dllexport) int _stdcall MyVNAGetScanAverage(int *pnAverage)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNAGetHelper( _T("ScanAverage"), VT_I4, pnAverage );
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// get and set miscellaneous integer array based data
__declspec(dllexport) int _stdcall MyVNAGetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnResult)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNAGetIntArrayHelper( _T("GetIntegerArray"), nWhat, nIndex, nArraySize, pnResult );
}

__declspec(dllexport) int _stdcall MyVNASetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnData)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNASetIntArrayHelper( _T("SetIntegerArray"), nWhat, nIndex, nArraySize, pnData );
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// get and set miscellaneous double array based data
__declspec(dllexport) int _stdcall MyVNAGetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnResult)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNAGetDoubleArrayHelper( _T("GetDoubleArray"), nWhat, nIndex, nArraySize, pnResult );
}

__declspec(dllexport) int _stdcall MyVNASetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnData)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNASetDoubleArrayHelper( _T("SetDoubleArray"), nWhat, nIndex, nArraySize, pnData );
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// get and set miscellaneous double array based data
__declspec(dllexport) int _stdcall MyVNASetString(int nWhat, int nIndex, _TCHAR * sValue )
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNASetStringHelper( _T("SetString"), nWhat, nIndex, sValue );
}

__declspec(dllexport) CString _stdcall MyVNAGetString(int nWhat, int nIndex )
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNAGetStringHelper( _T("GetString"), nWhat, nIndex );
}


// /////////////////////////////////////////////////////////////////////////////////////////////////
// windows positioning and control

__declspec(dllexport) int _stdcall MyVNAShowWindow(int nValue)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	static BYTE BASED_CODE parms[] =	VTS_I4;
	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)_T("ShowWindow"));
	if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
		if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
		{
			theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_EMPTY, NULL, parms, nValue, 0);
			errcode = 0;
		}
	return errcode;
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// frequency set / get functions

// this first one sets start & end and also sets frequency control mode.

__declspec(dllexport) int _stdcall MyVNASetFequencies(double f1, double f2, int nFlags)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	static BYTE BASED_CODE parms[] =	VTS_R8 VTS_R8 VTS_I4;
	int errcode = 1;
	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)_T("ScanFreqs"));
	if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
		if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
		{
			theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_EMPTY, NULL, parms, f1, f2, nFlags);
			errcode = 0;
		}
	return errcode;
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// Get results

__declspec(dllexport) int _stdcall MyVNAGetScanData(int nStart, int nEnd, int nWhata, int nWhatb, double *pDataA, double *pDataB )
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	int errcode = 1;
	int count = nEnd-nStart+1;
	if( count < 1 )
		return -1;

	SAFEARRAYBOUND rgsabound;
	rgsabound.cElements = count;
	rgsabound.lLbound = 0;

	VARIANT B, A;
	B.vt = A.vt = VT_ARRAY | VT_R8;

	SAFEARRAY FAR *pBArray, *pAArray;

	pBArray = SafeArrayCreate(VT_R8, 1, &rgsabound);
	pAArray = SafeArrayCreate(VT_R8, 1, &rgsabound);

	B.parray = pBArray;
	A.parray = pAArray;

	static BYTE parms[] = VTS_I4 VTS_I4 VTS_I4 VTS_I4 VTS_PVARIANT VTS_PVARIANT;

	DISPID dispid;
	OLECHAR FAR* szMember = T2OLE((LPTSTR)_T("GetScanData"));
	if( theApp.m_autoMyVNAObject->m_lpDispatch != NULL || (errcode = MyVNAInit()) == 0 )
		if( theApp.m_autoMyVNAObject->m_lpDispatch->GetIDsOfNames(IID_NULL, &szMember, 1, LOCALE_SYSTEM_DEFAULT, &dispid) == S_OK )
			theApp.m_autoMyVNAObject->InvokeHelper(dispid, DISPATCH_METHOD, VT_I4, &errcode, parms, nStart, nEnd, nWhata, nWhatb, &A, &B );

	if( errcode == 0 )
	{
		double * pDataAtemp = (double *)A.parray->pvData;
		double * pDataBtemp = (double *)B.parray->pvData;
		if( nWhata >= -1 )
			for( int i=0; i<count; i++)
				pDataA[i] = pDataAtemp[i];
		if( nWhatb >= -1 )
			for( int i=0; i<count; i++)
				pDataB[i] = pDataBtemp[i];
	}
	if (pBArray != NULL )
		SafeArrayDestroy( pBArray );
	if (pAArray != NULL )
		SafeArrayDestroy( pAArray );

	return errcode;
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// misc functions - load & save various things

__declspec(dllexport) int _stdcall MyVNALoadConfiguration(_TCHAR * fileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionBstr( _T("LoadConfiguration"), fileName );
}

__declspec(dllexport) int _stdcall MyVNASaveConfiguration(_TCHAR * fileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionBstr( _T("SaveConfiguration"), fileName );
}

__declspec(dllexport) int _stdcall MyVNALoadCalibration(_TCHAR * fileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionBstr( _T("LoadCalibration"), fileName );
}

__declspec(dllexport) int _stdcall MyVNASaveCalibration(_TCHAR * fileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionBstr( _T("SaveCalibration"), fileName );
}

__declspec(dllexport) int _stdcall MyVNASaveTraceData( _TCHAR * fileName )
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionBstr( _T("SaveTraceData"), fileName );
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// misc other functions
__declspec(dllexport) int _stdcall MyVNAAutoscale()
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionVoid( _T("Autoscale") );
}

__declspec(dllexport) int _stdcall MyVNAClipboardCopy()
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionVoid( _T("ClipboardCopy") );
}

__declspec(dllexport) int _stdcall MyVNASetEqCctLogFile(_TCHAR * fileName)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionBstr( _T("SetEqCctLogFile"), fileName );
}

__declspec(dllexport) int _stdcall MyVNALogEqCctResults(_TCHAR * description)
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	return MyVNACallFunctionBstr( _T("EqCctLogResults"), description );
}


