package com.example.trial4;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.trial4.ml.Model6;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    TextView result, confidence;
    ImageView imageView;
    Button picture;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                    // Launch gallery picker if we have permission
                    Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
                    pickImageIntent.setType("image/*");
                    startActivityForResult(pickImageIntent, 1);
                } else {
                    // Request permission to access the gallery if we don't have it
                    requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
                }


            }
        });
    }


    public void classifyImage(Bitmap image){

        try {
            Model6 model = Model6.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, imageSize, 0, 0, imageSize, imageSize);
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int pixel = intValues[i * imageSize + j]; // Corrected line
                    byteBuffer.putFloat(((pixel >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((pixel >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((pixel & 0xFF) * (1.f / 255.f));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model6.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();



            float[] confidences = outputFeature0.getFloatArray();
            int maxPos =0;
            float maxConfidence =0;
            for (int i=0; i < confidences.length;i++ ){
                if (confidences[i] > maxConfidence){
                    maxConfidence =confidences[i];
                    maxPos = i;

                }
            }

            String[] classes = {"Amoeba", "Euglena", "Hydra", "Paramecium", "Rod bacteria", "spherical bacteria", "spiral bacteria", "yeast", "Out of Bounds"};

            result.setText(classes[maxPos]);

            String s = "";
            for (int i =0; i< classes.length; i++){
                s += String.format("%s. %.1f%%\n", classes[i],confidences[i] * 100);
            }
            confidence.setText(s);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

        //code to check if the result is "no chicken fecal images detected"
        if (result.equals("no chicken fecal images detected")) {
            Toast.makeText(this, "No chicken fecal images detected", Toast.LENGTH_SHORT).show();
        }

    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {

            // Get the selected image from the intent
            Uri selectedImage = data.getData();
            try {
                // Convert the image to a Bitmap
                Bitmap image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                // Set the bitmap in the ImageView preview
                imageView.setImageBitmap(image);
                // Pass the Bitmap to the classifyImage() method for processing
                classifyImage(image);

            } catch (IOException e) {
                e.printStackTrace();
            }

        } else if (requestCode == 2) {
            Bundle extras = data.getExtras();
            Bitmap image = (Bitmap) extras.get("data");
            classifyImage(image);
        }else if  (requestCode == 101) {
            // Permission granted, launch the camera activity
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takePictureIntent, 2);
            }
        } else {
            // Permission denied, show a message or do something else
            // ...
        }


    }
}