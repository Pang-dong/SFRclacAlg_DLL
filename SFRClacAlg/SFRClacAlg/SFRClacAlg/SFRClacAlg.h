// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 SFRCLACALG_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// SFRCLACALG_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef SFRCLACALG_EXPORTS
#define SFRCLACALG_API __declspec(dllexport)
#else
#define SFRCLACALG_API __declspec(dllimport)
#endif
enum SFRClac_ErrorCode {
	Clac_Success = 0x0,
	Image_BufferError = 0x100,
	Image_QualityError = 00101,
	SFR_VALUEERROR = 0x200,

};
// 此类是从 SFRClacAlg.dll 导出的
class SFRCLACALG_API CSFRClacAlg {
public:
	CSFRClacAlg(void);
	~CSFRClacAlg();
	/*
	input:
		img：是已经提取到的斜边ROI区域 
		imgW: 斜边ROI区域的宽
		imgH: 斜边ROI区域的高
		cyclepixel:频率 "0.25 4分频"
	output:	
		sfrVal:当下频率 SFR值
	*/
	static int EdgeSFR_HB(unsigned char* img, int imgW, int imgH, double cyclepixel, double& sfrVal);
	// TODO:  在此添加您的方法。
};
