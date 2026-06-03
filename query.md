request URL for SFIS:
Input data - query a single SN
http://10.52.1.9/SFIS/Production/Travelers/Trav_1/resources/getQueryJSON.jsp?ColItem=serial_number&Field_Kind=ALLFIELD&fdSerial_Number=Y&fdModel_Serial=Y&fdMo_Number=Y&fdLine_Name=Y&fdSection_Name=Y&fdGroup_Name=Y&fdStation_Name=Y&fdIn_Station_Time=Y&fdOut_Station_Time=Y&fdRETEST_SEQ=Y&fdEmp_No=Y&fdQa_NO=Y&fdQa_Result=Y&fdPallet_No=Y&fdCarton_NO=Y&fdPO_NO=Y&fdCONTAINER_NO=Y&fdSHIPPING_SN=Y&fdMAC=Y&fdTRACK_NO=Y&fdKEY_PART_NO=Y&fdMODEL_NAME=Y&fdBill_NO=Y&fdError_flag=Y&fdFinish_flag=Y&fdVersion_Code=Y&fdSpecial_route=Y&fdCust_model=Y&fdCust_PN=Y&fdInv_no=Y&fdOther_MaC=Y&fdKP_NO_C=Y&fdMain_Product=Y&fdProduct_Name=Y&fdBox_No=Y&fdLOTN=Y&fdLOTB=Y&fdOut_Line_Time=Y&fdDRYBOX=Y&fdVIRTUAL_LINE1=Y&fdVIRTUAL_LINE2=Y&fdPanelSeq=Y&fdBCadd=Y&fdBCqry=Y&InpData=HMHHT7009X50000LQ7&FromURL=N

2A defect - query from a period of time
http://10.52.1.9/SFIS/Yield/Manager_2A/resources/getQueryJSON.jsp?profitCenter=0000000025&projectVersion=ALL&fromDate=2026%2F05%2F12%2000%3A00&toDate=2026%2F05%2F12%2023%3A59&BU=&Customer=&buildEvent=&family=&buildConfig=&MO=ALL&modelSerial=&modelName=&lotNo=&bigLot=&testStation=ALL&lineName=&groupName=ALL&errorCode=&majorProject=&projectName=&productName=&retestSequence=FIRST&recordType=ALL&empNo=ALL&processType=ALL&cbxGroupName=Yes&cbxTestTime=Yes&cbxErrorCode=Yes&group_name=ALL&mo_number=ALL

Query by model name
http://10.52.1.9/SFIS/PVS-vs-SFIS/SN/resources/getQuery.jsp
Payload:
fromDate
2026/05/11 
toDate
2026/05/12 
disable_period
on
buildevent
modelname
family
sn
HMHHL400B0V0000LQ7
config
comppn
location
U7000
mo
carton_no
