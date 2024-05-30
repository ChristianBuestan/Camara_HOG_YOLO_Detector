package com.example.appcpp;


import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.FaceDetector;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoWriter;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class MainActivityCPP extends CameraActivity {
    private  String LOGTAG="OpenCV_log";
    private CameraBridgeViewBase mOpenCvCameraView;
    private int vidGG=0;

    static {
       System.loadLibrary("appcpp");
    }
    public native void loadONNXModel(AssetManager assetManager, String modelPath);
    public native void cargarLabels(AssetManager assetManager);
    public native void cargarHOG(AssetManager assetManager);

    public native void Detectar(Mat matOut);

    public native void detectHOG(Mat matOut);

    private int caso=0;
    private Mat input_rgba;

    private BaseLoaderCallback mLoaderCallback= new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                //este le importe, me mostrara si mi openCV fue cargado en base al backend de carga.
                case LoaderCallbackInterface.SUCCESS:{
                    mOpenCvCameraView.enableView();
                }break;
                default:{
                    super.onManagerConnected(status);
                }break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.extra);

        //Cargo el modelo y etiquetas, para usarlo en onCameraFrame
        loadONNXModel(getAssets(),"best.onnx");
        //Tambien puedo cargarlo sin el nombre, este ya esta en el cpp.
        cargarLabels(getAssets());
        //Cargo mi HOG de YML
        cargarHOG(getAssets());

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.javaCameraView2);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener);
        mOpenCvCameraView.setCameraIndex(0);

        Button grises=findViewById(R.id.Grises);
        grises.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                caso=1;
            }
        });
        Button bordes=findViewById(R.id.Bordes);
        bordes.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) { caso=2;}
        });
        Button reduc=findViewById(R.id.Ruido);
        reduc.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                caso=3;
            }
        });
    }
    @Override
    protected List<?extends  CameraBridgeViewBase> getCameraViewList(){
        return Collections.singletonList(mOpenCvCameraView);
    }
    private CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener = new CameraBridgeViewBase.CvCameraViewListener2() {
        @Override
        public void onCameraViewStarted(int width, int height) {

        }

        @Override
        public void onCameraViewStopped() {

        }
        //Este es donde grabo y muestro mis frames por lo que aqui es donde los modifico.
        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            input_rgba =inputFrame.rgba();
            //Transformo RGBA a RGB
            Imgproc.cvtColor(input_rgba, input_rgba, Imgproc.COLOR_RGBA2RGB);
            switch (caso){
                case 1:
                    Detectar(input_rgba);
                    break;
                case 2:
                    detectHOG(input_rgba);
                    break;
                case 3:
                    break;
                default:
                    break;
            }
            return input_rgba;//MAT
        }
    };
    @Override
    public void  onPause(){
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }
    @Override
    public void  onResume(){
        super.onResume();
        if (! OpenCVLoader.initDebug()){
            Log.d(LOGTAG,"OpenCV no encontrado, iniciandolo");
            //ASincronizo
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this,mLoaderCallback);
        }else{
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    //Cuando cierro mi app que se termine la grabacion de video.
    @Override
    public void  onDestroy(){
        super.onDestroy();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

}