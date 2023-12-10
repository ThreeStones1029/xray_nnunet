#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkResampleImageFilter.h>
#include <itkCenteredEuler3DTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkGDCMImageIO.h>
#include <itkPNGImageIO.h>
#include <itkPNGImageIOFactory.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkImageSeriesWriter.h>
#include <itkNumericSeriesFileNames.h>
#include <itkGiplImageIOFactory.h>
#include <itkCastImageFilter.h>
#include <itkNiftiImageIO.h>
#include <itkNiftiImageIOFactory.h>
#include <itkMetaImageIOFactory.h>
#include <itkRayCastInterpolateImageFunction.h>
#include <iostream>
#include <sys/io.h>


using namespace std;

extern "C"{
	int Generate_drr(float rx_value, float ry_value, float rz_value,
		float tx_value, float ty_value, float tz_value,
		float cx_value, float cy_value, float cz_value,
		float sid_value, 
		float sx_value, float sy_value, 
		int dx_value, int dy_value, 
		float o2Dx_value, float o2Dy_value,
		double threshold_value,
		const char* ct_file_path,
		const char* drr_save_path
		) {
		

		// 绕X、Y和Z轴的旋转角度（以度为单位）
		float rx = rx_value;
		float ry = ry_value;
		float rz = rz_value;

		// 三维平移向量的分量
		float tx = tx_value;
		float ty = ty_value;
		float tz = tz_value;

		// 旋转中心的坐标
		float cx = cx_value;
		float cy = cy_value;
		float cz = cz_value;

		// 源到成像平面的距离（即源与检测器之间的距离）
		float sid = sid_value;

		// DRR图像在X和Y轴上的像素间隔
		float sx = sx_value;
		float sy = sy_value;

		// DRR图像的宽度和高度（以像素为单位）
		int dx = dx_value;
		int dy = dy_value;

		// DRR图像的原点偏移量
		float o2Dx = o2Dx_value;
		float o2Dy = o2Dy_value;

		// 图像阈值
		// 较低的阈值：
		//	将保留更多原始 CT 数据中的像素，包括低密度区域。
		//	这可能导致生成的 DRR 图像中显示更多的细节，但也可能包含一些噪音或无关信息。
		// 较高的阈值：
		//	将过滤掉原始 CT 数据中的低密度区域，只保留高密度区域。
		//	这可能导致生成的 DRR 图像中显示更强的骨骼结构，但可能丢失一些低密度的软组织信息。
		float threshold = threshold_value;

		// 你自己的CT文件的nii路径
		const char* file_path = ct_file_path; //ct路径
		
		const char* save_path = drr_save_path;

		cout << "input_path = " << file_path << endl;
		cout << "save_path = " << save_path << endl;

		bool ok;
		bool verbose = false;

		const unsigned int Dimension = 3;
		using InputPixelType = float;
		using OutputPixelType = unsigned char;
		using InputImageType = itk::Image<InputPixelType, Dimension>;
		using OutputImageType = itk::Image<OutputPixelType, Dimension>;

		InputImageType::Pointer image;
		
		using ReaderType = itk::ImageFileReader<InputImageType>;
		ReaderType::Pointer reader = ReaderType::New();
		
		
		
		itk::NiftiImageIOFactory::RegisterOneFactory();
		reader->SetFileName(file_path);
		try {
			reader->Update();
		}
		catch (itk::ExceptionObject& err) {
			cerr << "error exception object caught!" << endl;
			cerr << err.what() << endl;
			return EXIT_FAILURE;
		}
		image = reader->GetOutput();

		if (verbose) {
			const InputImageType::SpacingType spacing = image->GetSpacing();
			cout << endl << "Input: ";

			InputImageType::RegionType region = image->GetBufferedRegion();
			region.Print(cout);

			cout << " Resolution: [";
			for (int i = 0; i < Dimension; i++) {
				cout << spacing[i];
				if (i < Dimension - 1) cout << ", ";
			}
			cout << "]" << endl;

			const InputImageType::PointType origin = image->GetOrigin();
			cout << " Origin: [";
			for (int i = 0; i < Dimension; i++) {
				cout << origin[i];
				if (i < Dimension - 1) cout << ", ";
			}
			cout << "]" << endl << endl;
		}


		using FilterType = itk::ResampleImageFilter<InputImageType, InputImageType>;
		FilterType::Pointer filter = FilterType::New();
		filter->SetInput(image);
		filter->SetDefaultPixelValue(0);

		using TransformType = itk::CenteredEuler3DTransform<double>;
		TransformType::Pointer transform = TransformType::New();

		transform->SetComputeZYX(true);
		TransformType::OutputVectorType translation;

		translation[0] = tx;
		translation[1] = ty;
		translation[2] = tz;

		const double dtr = atan(1.0) * 4.0 / 180.0;

		transform->SetTranslation(translation);
		transform->SetRotation(dtr * rx, dtr * ry, dtr * rz);

		InputImageType::PointType imOrigin = image->GetOrigin();
		InputImageType::SpacingType imRes = image->GetSpacing();

		using InputImageRegionType = InputImageType::RegionType;
		using InputImageSizeType = InputImageRegionType::SizeType;

		InputImageRegionType imRegion = image->GetBufferedRegion();
		InputImageSizeType imSize = imRegion.GetSize();

		imOrigin[0] += imRes[0] * static_cast<double>(imSize[0]) / 2.0;
		imOrigin[1] += imRes[1] * static_cast<double>(imSize[1]) / 2.0;
		imOrigin[2] += imRes[2] * static_cast<double>(imSize[2]) / 2.0;
		
		TransformType::InputPointType center;
		center[0] = cx + imOrigin[0];
		center[1] = cy + imOrigin[1];
		center[2] = cz + imOrigin[2];
		transform->SetCenter(center);

		if (verbose) {
			cout << "Image size: " << imSize[0] << ", " << imSize[1] << ", " << imSize[2]
				<< endl << " resolution: " << imRes[0] << ", " << imRes[1] << ", " << imRes[2]
				<< endl << " origin: " << imOrigin[0] << ", " << imOrigin[1] << ", " <<
				imOrigin[2] << endl << " center: " << center[0] << ", " << center[1]
				<< ", " << center[2] << endl << "Transform: " << transform << endl;
		}
		using InterpolatorType = itk::RayCastInterpolateImageFunction<InputImageType, double>;
		InterpolatorType::Pointer interpolator = InterpolatorType::New();
		interpolator->SetTransform(transform);

		interpolator->SetThreshold(threshold);
		InterpolatorType::InputPointType focalpoint;

		focalpoint[0] = imOrigin[0];
		focalpoint[1] = imOrigin[1];
		focalpoint[2] = imOrigin[2] - sid / 2.0;

		interpolator->SetFocalPoint(focalpoint);

		if (verbose) {
			cout << "Focal Point: "
				<< focalpoint[0] << ", "
				<< focalpoint[1] << ", "
				<< focalpoint[2] << endl;
		}
		interpolator->Print(std::cout);

		filter->SetInterpolator(interpolator);
		filter->SetTransform(transform);

		// setup the scene
		InputImageType::SizeType   size;
		size[0] = dx;  // number of pixels along X of the 2D DRR image
		size[1] = dy;  // number of pixels along Y of the 2D DRR image
		size[2] = 1;   // only one slice

		filter->SetSize(size);

		InputImageType::SpacingType spacing;

		spacing[0] = sx;  // pixel spacing along X of the 2D DRR image [mm]
		spacing[1] = sy;  // pixel spacing along Y of the 2D DRR image [mm]
		spacing[2] = 1.0; // slice thickness of the 2D DRR image [mm]
		filter->SetOutputSpacing(spacing);

		if (verbose)
		{
			std::cout << "Output image size: "
				<< size[0] << ", "
				<< size[1] << ", "
				<< size[2] << std::endl;

			std::cout << "Output image spacing: "
				<< spacing[0] << ", "
				<< spacing[1] << ", "
				<< spacing[2] << std::endl;
		}

		double origin[Dimension];
		origin[0] = imOrigin[0] + o2Dx - sx * ((double)dx - 1.) / 2.;
		origin[1] = imOrigin[1] + o2Dy - sy * ((double)dy - 1.) / 2.;
		origin[2] = imOrigin[2] + sid / 2.;
		filter->SetOutputOrigin(origin);
		if (verbose)
		{
			std::cout << "Output image origin: "
				<< origin[0] << ", "
				<< origin[1] << ", "
				<< origin[2] << std::endl;
		}

		// create writer
		using RescaleFilterType = itk::RescaleIntensityImageFilter<InputImageType, OutputImageType>;
		RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
		rescaler->SetOutputMinimum(0);
		rescaler->SetOutputMaximum(255);
		rescaler->SetInput(filter->GetOutput());

		using WriterType = itk::ImageFileWriter<OutputImageType>;
		WriterType::Pointer writer = WriterType::New();

		using pngType = itk::PNGImageIO;
		pngType::Pointer pngIO1 = pngType::New();
		itk::PNGImageIOFactory::RegisterOneFactory();
		writer->SetFileName(save_path);
		writer->SetImageIO(pngIO1);
		writer->SetImageIO(itk::PNGImageIO::New());
		writer->SetInput(rescaler->GetOutput());
		try
		{
			std::cout << "Writing image: " << save_path << std::endl;
			writer->Update();
		}
		catch (itk::ExceptionObject& err)
		{
			std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
		}


		return 0;
	}


int main(int argc, char* argv[]) {

	// 绕X、Y和Z轴的旋转角度（以度为单位）
	float rx = -90;
	float ry = 0;
	float rz = 0;

	// 三维平移向量的分量
	float tx = 0;
	float ty = 0;
	float tz = 0;

	// 旋转中心的坐标
	float cx = 0;
	float cy = 0;
	float cz = 0;

	// 源到成像平面的距离（即源与检测器之间的距离）
	float sid = 1000;

	// DRR图像在X和Y轴上的像素间隔
	float sx = 0.25;
	float sy = 0.25;

	// DRR图像的宽度和高度（以像素为单位）
	int dx = 1000;
	int dy = 1000;

	// DRR图像的原点偏移量
	float o2Dx = 0;
	float o2Dy = 0;

	// 图像阈值
	// 较低的阈值：
	//	将保留更多原始 CT 数据中的像素，包括低密度区域。
	//	这可能导致生成的 DRR 图像中显示更多的细节，但也可能包含一些噪音或无关信息。
	// 较高的阈值：
	//	将过滤掉原始 CT 数据中的低密度区域，只保留高密度区域。
	//	这可能导致生成的 DRR 图像中显示更强的骨骼结构，但可能丢失一些低密度的软组织信息。
	float threshold = 0;

	// CT文件位置，要求.nii.gz
	const char* ct_file_path = "";

	// DRR保存位置，后缀.png
	const char* drr_save_path = "";

	// 检查命令行参数数量是否正确
	if (argc != 20) {
		std::cout << "错误的参数数量！" << std::endl;
		return 1;
	}

	// 逐个读取命令行参数并赋值给对应的变量
	rx = atof(argv[1]);
	ry = atof(argv[2]);
	rz = atof(argv[3]);
	tx = atof(argv[4]);
	ty = atof(argv[5]);
	tz = atof(argv[6]);
	cx = atof(argv[7]);
	cy = atof(argv[8]);
	cz = atof(argv[9]);
	sid = atof(argv[10]);
	sx = atof(argv[11]);
	sy = atof(argv[12]);
	dx = atoi(argv[13]);
	dy = atoi(argv[14]);
	o2Dx = atof(argv[15]);
	o2Dy = atof(argv[16]);
	threshold = atof(argv[17]);
	ct_file_path = argv[18];
	drr_save_path = argv[19];

	Generate_drr(rx, ry, rz, tx, ty, tz, cx, cy, cz, sid, sx, sy, dx, dy, o2Dx, o2Dy, threshold, ct_file_path, drr_save_path);
	return 0;
}
}