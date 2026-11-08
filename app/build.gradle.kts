plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.sun1423.playtest"
    compileSdk = 35
    ndkVersion = "28.2.13676358"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.sun1423.playtest.v2"
        minSdk = 28
        targetSdk = 35
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        versionName = "2.1." + (System.getenv("GITHUB_RUN_NUMBER") ?: "1")

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DQVAC_TTS_SOURCE_DIR=${rootProject.projectDir}/third_party/qvac-fabric-speech.cpp/engines/tts"
                )
                cppFlags += listOf("-std=c++17")
            }
        }
    }

    signingConfigs {
        create("playtest") {
            val keystorePath = System.getenv("PLAYTEST_KEYSTORE_PATH")
                ?: throw GradleException("PLAYTEST_KEYSTORE_PATH is not set")
            storeFile = file(keystorePath)
            storePassword = System.getenv("PLAYTEST_KEYSTORE_PASSWORD")
                ?: throw GradleException("PLAYTEST_KEYSTORE_PASSWORD is not set")
            keyAlias = "playtest"
            keyPassword = System.getenv("PLAYTEST_KEYSTORE_PASSWORD")
            storeType = "PKCS12"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("playtest")
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("playtest")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.0"
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.10.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.media3:media3-exoplayer:1.10.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
