// Write C++ code here.
#include <jni.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>
#include <android/log.h>
using namespace cv;
using namespace cv::dnn;
using namespace std;

cv::dnn::Net neuralNetwork;
std::vector<std::string> labels;
const float IMG_WIDTH = 640.0;
const float IMG_HEIGHT = 640.0;
//Valores threshold(Ver canny), para las detecciones de objetos.
const float CLASS_PROBABILITY = 0.1;
const float NMS_THRESHOLD = 0.1;
const float CONFIDENCE_THRESHOLD = 0.1;
const int NUMBER_OF_OUTPUTS = 13; // Number of variables for each detection
// Text parameters. Estos dos ultimos son Mis fonts y sus formas
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 13;
// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

//HOG LOCAL
cv::HOGDescriptor hog;


extern "C" JNIEXPORT void JNICALL
Java_com_example_appcpp_MainActivityCPP_loadONNXModel(JNIEnv *env, jobject /* this */,
                                                      jobject assetManager, jstring modelPath) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    const char *modelPathStr = env->GetStringUTFChars(modelPath, nullptr);

    AAsset *modelAsset = AAssetManager_open(mgr, modelPathStr, AASSET_MODE_BUFFER);
    env->ReleaseStringUTFChars(modelPath, modelPathStr);

    // Retorna el puntero a tu modelo cargado.
    if (modelAsset != nullptr) {
        off_t modelSize = AAsset_getLength(modelAsset);
        const void *modelData = AAsset_getBuffer(modelAsset);

        // Convertimos el puntero void* a uchar* antes de crear el vector
        const uchar* modelDataUChar = static_cast<const uchar*>(modelData);
        std::vector<uchar> modelBuffer(modelDataUChar, modelDataUChar + modelSize);

        AAsset_close(modelAsset);

        cv::Mat modelMat(modelBuffer);
        neuralNetwork = cv::dnn::readNetFromONNX(modelMat);
        //Probar con tflite

        // Verificamos si el modelo se cargó correctamente
        if (!neuralNetwork.empty()) {
            // El modelo se cargó correctamente, puedes imprimir un mensaje
            __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "Modelo cargado correctamente.");
           /* vector<string> nombresCapas = neuralNetwork.getUnconnectedOutLayersNames();
            for (string nombre : nombresCapas) {
                __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "capa salida: %s",nombre.c_str());
            }*/

        } else {
            // Hubo un problema al cargar el modelo
            __android_log_print(ANDROID_LOG_ERROR, "REGISTRO", "Error al cargar el modelo.");
        }
    } else {
        // No se pudo abrir el archivo del modelo
        __android_log_print(ANDROID_LOG_ERROR, "REGISTRO", "Error al cargar el modelo.");
    }
}

//Cargar coco.txt.
std::vector<std::string> loadLabelsCOCO(JNIEnv *env, jobject assetManager, const char *fileName, char sep = '\n') {
    std::vector<std::string> names;
    // Convertir el assetManager de Java a AAssetManager nativo de Android
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    // Abrir el archivo desde la carpeta assets
    AAsset *asset = AAssetManager_open(mgr, fileName, AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        // Manejar el error si no se pudo abrir el archivo
        // Puedes imprimir un mensaje de error o lanzar una excepción según tu preferencia.
        return names;
    }
    // Obtener el tamaño del archivo
    off_t fileSize = AAsset_getLength(asset);
    // Leer el contenido del archivo en un buffer
    std::vector<char> fileContent(fileSize);
    AAsset_read(asset, fileContent.data(), fileSize);
    // Cerrar el archivo después de leer
    AAsset_close(asset);
    // Procesar el contenido del archivo
    std::string content(fileContent.begin(), fileContent.end());
    std::istringstream buffer(content);
    std::string token;
    while (getline(buffer, token, sep)) {
        if (!token.empty()) {
            names.push_back(token);
        }
    }
    return names;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_appcpp_MainActivityCPP_cargarLabels(JNIEnv *env, jobject /* this */, jobject assetManager) {
    //nombre del archivo
    const char *fileName = "obj.names";

    // Llama a la función para cargar las etiquetas
    labels = loadLabelsCOCO(env, assetManager, fileName);
    if (!labels.empty()){
        __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "Etiquetas Cargadas.");
    }else{
        __android_log_print(ANDROID_LOG_ERROR, "REGISTRO", "Error de carga de etiquetas.");
    }
}
//USO DE LABELS Y NETWORK
vector<Mat> forwardNET(Mat inputImage, Net net) {
    try {
        //int numCanales = inputImage.channels();
        // Crear un blob a partir de la imagen de entrada
        Mat blob;
        blobFromImage(inputImage, blob, 1. / 255., Size(IMG_WIDTH,
                                        IMG_HEIGHT), Scalar(), true, false);
        // Establecer el blob como entrada para la red neuronal
        net.setInput(blob);
        // Realizar la propagación hacia adelante (forward pass)
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Devolver los resultados de la detección
        return outputs;
    } catch (const cv::Exception& e) {
        // Capturar la excepción y obtener el mensaje de error
        const char* errorMsg = e.what();

        // Imprimir el mensaje de error en Logcat
        __android_log_print(ANDROID_LOG_ERROR, "REGISTRO", "Error en forwardNET: %s", errorMsg);

        // Puedes manejar el error de alguna otra manera si es necesario
        // Por ejemplo, devolver un vector vacío o lanzar una nueva excepción
        return vector<Mat>();  // Vector vacío en caso de error
    }
}
void draw_label(Mat input_image, string label, int left, int top) {
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    Point tlc = Point(left, top);
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE,
            FONT_SCALE, YELLOW, THICKNESS);
}

Mat filterDetections(Mat inputImg, vector<Mat> detections, vector<string> classNames) {
    Mat inputImage = inputImg.clone(); //Clono mi imagen
    vector<int> classIDs;  //identificadores de clases
    vector<float> confidences; //Niveles de confianzas
    vector<Rect> boxes; // Rectangulos de detecciones
    // Resizing factor.
    float x_factor = inputImage.cols / IMG_WIDTH;
    float y_factor = inputImage.rows / IMG_HEIGHT;
    float* pData = new float[NUMBER_OF_OUTPUTS]; // = (float *)detections[0].data;
    float confidence = 0.0;
    float* probValues;
    Point classId;
    double maxClassProb = 0.0;
    Mat probabilityClasses = Mat::zeros(1, classNames.size(), CV_32FC1);
    int totalDetections = detections[0].total() / NUMBER_OF_OUTPUTS;
    for (int i = 0; i < totalDetections; ++i) {
        std::memcpy(pData, (float*)detections[0].data + (i * NUMBER_OF_OUTPUTS), NUMBER_OF_OUTPUTS * sizeof(float));
        confidence = pData[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            probValues = (pData + 5);
            probabilityClasses = Mat::zeros(1, classNames.size(), CV_32FC1);
            std::memcpy(probabilityClasses.data, probValues, classNames.size() * sizeof(float));
            minMaxLoc(probabilityClasses, 0, &maxClassProb, 0, &classId);
            if (maxClassProb > CLASS_PROBABILITY) {
                confidences.push_back(confidence);
                classIDs.push_back(classId.x);
                boxes.push_back(Rect(int((pData[0] - 0.5 * pData[2]) * x_factor), int((pData[1] - 0.5 * pData[3]) * y_factor),
                                     int(pData[2] * x_factor), int(pData[3] * y_factor)));

            }
        }
    }
    vector<int> indices;
    string label = "";
    NMSBoxes(boxes, confidences, CLASS_PROBABILITY, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) {
        rectangle(inputImage, boxes[indices[i]], BLUE, 3 * THICKNESS);
        label = format("%.2f", confidences[indices[i]]);
        label = classNames[classIDs[indices[i]]] + ":" + label;

        draw_label(inputImage, label, boxes[indices[i]].x, boxes[indices[i]].y);
    }

    return inputImage;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_appcpp_MainActivityCPP_Detectar(
        JNIEnv* env,
        jobject /* this */,
        jobject matOut){
    jclass matClass = env->GetObjectClass(matOut);
    jmethodID getNativeObjAddrMethod = env->GetMethodID(matClass, "getNativeObjAddr", "()J");
    jlong matAddr = env->CallLongMethod(matOut, getNativeObjAddrMethod);
    cv::Mat* mat = reinterpret_cast<cv::Mat*>(matAddr);
    int mat_width=mat->cols;
    int mat_heigh=mat->rows;
    Mat detectar;
    mat->copyTo(detectar);
    if(!neuralNetwork.empty()){
        vector<Mat> detections= forwardNET(detectar,neuralNetwork);
        detectar= filterDetections(detectar,detections,labels);
        vector<double> layersTimes;
        //Veo cuanto tarde en hacer la deteccion y la grafico en mi imagen.
        double freq = getTickFrequency() / 1000;
        double t = neuralNetwork.getPerfProfile(layersTimes) / freq;
        string label = format("Time : %.2f ms", t);
        putText(detectar, label, Point(240, 40), FONT_FACE, FONT_SCALE, RED);
        resize(detectar,detectar,Size(mat_width,mat_heigh));
    }
    detectar.copyTo(*mat);
    env->DeleteLocalRef(matClass);
}
// Enumeración para mapear el valor entero a HistogramNormType
enum class HistogramNormTypeInt {
    L2Hys = 0,
    L2HysGrad = 1,
    L1Hys = 2,
    L1HysGrad = 3
};

void configurarHOGDesdeYAML(const std::string& ovinosSection) {
    // Crear un flujo de entrada para procesar la sección 'ovinos'
    std::istringstream ovinosStream(ovinosSection);
    std::string key;
    char colon;

    // Leer los campos y valores de la sección 'ovinos'
    while (ovinosStream >> key >> colon) {
        if (key == "winSize:") {
            cv::Size winSize;
            ovinosStream >> winSize.width >> winSize.height;
            hog.winSize = winSize;
        } else if (key == "blockSize:") {
            cv::Size blockSize;
            ovinosStream >> blockSize.width >> blockSize.height;
            hog.blockSize = blockSize;
        } else if (key == "blockStride:") {
            cv::Size blockStride;
            ovinosStream >> blockStride.width >> blockStride.height;
            hog.blockStride = blockStride;
        } else if (key == "cellSize:") {
            cv::Size cellSize;
            ovinosStream >> cellSize.width >> cellSize.height;
            hog.cellSize = cellSize;
        } else if (key == "nbins:") {
            ovinosStream >> hog.nbins;
        } else if (key == "derivAperture:") {
            ovinosStream >> hog.derivAperture;
        } else if (key == "winSigma:") {
            ovinosStream >> hog.winSigma;
        }  else if (key == "histogramNormType:") {
            int histogramNormTypeInt;
            ovinosStream >> histogramNormTypeInt;
            // Convertir el valor entero a HistogramNormType
            hog.histogramNormType = static_cast<cv::HOGDescriptor::HistogramNormType>(
                    static_cast<HistogramNormTypeInt>(histogramNormTypeInt)
            );
        }else if (key == "L2HysThreshold:") {
            ovinosStream >> hog.L2HysThreshold;
        } else if (key == "gammaCorrection:") {
            ovinosStream >> hog.gammaCorrection;
        } else if (key == "nlevels:") {
            ovinosStream >> hog.nlevels;
        } else if (key == "signedGradient:") {
            ovinosStream >> hog.signedGradient;
        } else if (key == "SVMDetector:") {
            std::vector<float> svmDetectorValues;
            // Leer valores hasta que se alcance el final de la sección 'ovinos'
            while (ovinosStream >> std::ws, ovinosStream.peek() != '-' && ovinosStream.peek() != EOF) {
                float value;
                if (!(ovinosStream >> value)) {
                    __android_log_print(ANDROID_LOG_ERROR, "REGISTRO", "Error al leer 'SVMDetector'.");
                    return;  // Puedes agregar un valor de retorno o lanzar una excepción si es apropiado
                }
                svmDetectorValues.push_back(value);
            }
            // Asignar el vector SVMDetector al objeto HOG
            hog.setSVMDetector(svmDetectorValues);
        }
    }
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "Configuración de HOG completada.");
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "HOGDescriptor Configuration:");
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "winSize: %d, %d", hog.winSize.width, hog.winSize.height);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "blockSize: %d, %d", hog.blockSize.width, hog.blockSize.height);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "blockStride: %d, %d", hog.blockStride.width, hog.blockStride.height);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "cellSize: %d, %d", hog.cellSize.width, hog.cellSize.height);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "nbins: %d", hog.nbins);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "derivAperture: %d", hog.derivAperture);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "winSigma: %f", hog.winSigma);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "histogramNormType: %d", hog.histogramNormType);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "L2HysThreshold: %f", hog.L2HysThreshold);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "gammaCorrection: %d", hog.gammaCorrection);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "nlevels: %d", hog.nlevels);
    __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "signedGradient: %d", hog.signedGradient);

}

void procesarArchivoYAML(const char* buffer, off_t fileSize) {
    std::string yamlContent(buffer, fileSize);
    // Buscar la línea que comienza con "ovinos:"
    std::istringstream yamlStream(yamlContent);
    std::string line;
    while (std::getline(yamlStream, line)) {
        if (line.find("ovinos:") == 0) {
            // Encontramos la línea "ovinos:", ahora procesamos las líneas subsiguientes hasta encontrar "---"
            std::ostringstream ovinosSectionStream;
            while (std::getline(yamlStream, line)) {
                if (line.find("---") == 0) {
                    // Se encontró el final de la sección
                    break;
                }
                ovinosSectionStream << line << "\n";
            }
            // Extraer la sección "ovinos"
            std::string ovinosSection = ovinosSectionStream.str();
            __android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "Sección 'ovinos': %s", ovinosSection.c_str());
            // Procesar la sección 'ovinos' y configurar el objeto HOG
            configurarHOGDesdeYAML(ovinosSection);
            return;
        }
    }
    __android_log_print(ANDROID_LOG_ERROR, "REGISTRO", "No se encontró la sección 'ovinos'.");
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_appcpp_MainActivityCPP_cargarHOG(JNIEnv *env, jobject /* this */, jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    const char *filename = "ovinos.yml";

    AAsset *asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);

    // Obtener el tamaño del archivo
    off_t fileSize = AAsset_getLength(asset);

    // Leer el contenido del archivo YAML
    char *buffer = new char[fileSize];
    AAsset_read(asset, buffer, fileSize);
    AAsset_close(asset);

    procesarArchivoYAML(buffer, fileSize);

    delete[] buffer;
}
//El proceso anterior define y guarda mi HOG local, ahora lo utilizare.
extern "C" JNIEXPORT void JNICALL
Java_com_example_appcpp_MainActivityCPP_detectHOG(JNIEnv *env, jobject /* this */, jobject imagen) {
    jclass matClass = env->GetObjectClass(imagen);
    jmethodID getNativeObjAddrMethod = env->GetMethodID(matClass, "getNativeObjAddr", "()J");
    jlong matAddr = env->CallLongMethod(imagen, getNativeObjAddrMethod);
    cv::Mat* mat = reinterpret_cast<cv::Mat*>(matAddr);
    cv::Mat mat2; //This mat will be used for the detection
    cv::cvtColor(*mat, mat2, COLOR_RGB2GRAY);
    vector<Rect> deteccionesR;
    // Obtener la hora de inicio
    auto start_time = std::chrono::high_resolution_clock::now();

    hog.detectMultiScale(mat2, deteccionesR,
                         0.35, Size(16, 16), Size(64, 64),
                         3.04, 1.5, false);

    // Obtener la hora de finalización y dibujarla en la matriz
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::string tiempo_str = "Time: " + std::to_string(duration.count()) + " ms";
    putText(*mat, tiempo_str, Point(240, 40), FONT_FACE, FONT_SCALE, RED);

    // Dibujar rectángulos alrededor de las detecciones.
    cv::Rect rec;
    for (int i = 0; i < deteccionesR.size(); i++) {
        cout << "graficando" << endl;
        rec = deteccionesR[i];
        rectangle(*mat , rec, Scalar(233, 3, 3), 10);
    }
    //__android_log_print(ANDROID_LOG_DEBUG, "REGISTRO", "N",deteccionesR.size());
    env->DeleteLocalRef(matClass);
}