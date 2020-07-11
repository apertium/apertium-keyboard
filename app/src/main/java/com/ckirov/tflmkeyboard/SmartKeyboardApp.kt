package com.ckirov.tflmkeyboard

import android.app.Application
//import com.ckirov.tflmkeyboard.di.predictModule
//import org.koin.android.ext.android.startKoin

class SmartKeyboardApp : Application() {

    override fun onCreate() {
        super.onCreate()
        // start Koin!
        //startKoin(this, listOf(predictModule))
    }
}
