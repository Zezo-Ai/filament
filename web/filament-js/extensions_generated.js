// This file has been generated by beamsplitter

Filament.loadGeneratedExtensions = function() {

    Filament.View.prototype.setDynamicResolutionOptionsDefaults = function(overrides) {
        const options = {
            minScale: [0.5, 0.5],
            maxScale: [1.0, 1.0],
            sharpness: 0.9,
            enabled: false,
            homogeneousScaling: false,
            quality: Filament.View$QualityLevel.LOW,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setBloomOptionsDefaults = function(overrides) {
        const options = {
            // JavaScript binding for dirt is not yet supported, must use default value.
            // JavaScript binding for dirtStrength is not yet supported, must use default value.
            strength: 0.10,
            resolution: 384,
            levels: 6,
            blendMode: Filament.View$BloomOptions$BlendMode.ADD,
            threshold: true,
            enabled: false,
            highlight: 1000.0,
            quality: Filament.View$QualityLevel.LOW,
            lensFlare: false,
            starburst: true,
            chromaticAberration: 0.005,
            ghostCount: 4,
            ghostSpacing: 0.6,
            ghostThreshold: 10.0,
            haloThickness: 0.1,
            haloRadius: 0.4,
            haloThreshold: 10.0,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setFogOptionsDefaults = function(overrides) {
        const options = {
            distance: 0.0,
            cutOffDistance: Infinity,
            maximumOpacity: 1.0,
            height: 0.0,
            heightFalloff: 1.0,
            color: [ 1.0, 1.0, 1.0 ],
            density: 0.1,
            inScatteringStart: 0.0,
            inScatteringSize: -1.0,
            fogColorFromIbl: false,
            // JavaScript binding for skyColor is not yet supported, must use default value.
            enabled: false,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setDepthOfFieldOptionsDefaults = function(overrides) {
        const options = {
            cocScale: 1.0,
            cocAspectRatio: 1.0,
            maxApertureDiameter: 0.01,
            enabled: false,
            filter: Filament.View$DepthOfFieldOptions$Filter.MEDIAN,
            nativeResolution: false,
            foregroundRingCount: 0,
            backgroundRingCount: 0,
            fastGatherRingCount: 0,
            maxForegroundCOC: 0,
            maxBackgroundCOC: 0,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setVignetteOptionsDefaults = function(overrides) {
        const options = {
            midPoint: 0.5,
            roundness: 0.5,
            feather: 0.5,
            color: [0.0, 0.0, 0.0, 1.0],
            enabled: false,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setRenderQualityDefaults = function(overrides) {
        const options = {
            hdrColorBuffer: Filament.View$QualityLevel.HIGH,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setSsctDefaults = function(overrides) {
        const options = {
            lightConeRad: 1.0,
            shadowDistance: 0.3,
            contactDistanceMax: 1.0,
            intensity: 0.8,
            lightDirection: [ 0, -1, 0 ],
            depthBias: 0.01,
            depthSlopeBias: 0.01,
            sampleCount: 4,
            rayCount: 1,
            enabled: false,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setGtaoDefaults = function(overrides) {
        const options = {
            sampleSliceCount: 4,
            sampleStepsPerSlice: 3,
            thicknessHeuristic: 0.004,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setAmbientOcclusionOptionsDefaults = function(overrides) {
        const options = {
            aoType: Filament.View$AmbientOcclusionOptions$AmbientOcclusionType.SAO,
            radius: 0.3,
            power: 1.0,
            bias: 0.0005,
            resolution: 0.5,
            intensity: 1.0,
            bilateralThreshold: 0.05,
            quality: Filament.View$QualityLevel.LOW,
            lowPassFilter: Filament.View$QualityLevel.MEDIUM,
            upsampling: Filament.View$QualityLevel.LOW,
            enabled: false,
            bentNormals: false,
            minHorizonAngleRad: 0.0,
            // JavaScript binding for ssct is not yet supported, must use default value.
            // JavaScript binding for gtao is not yet supported, must use default value.
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setMultiSampleAntiAliasingOptionsDefaults = function(overrides) {
        const options = {
            enabled: false,
            sampleCount: 4,
            customResolve: false,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setTemporalAntiAliasingOptionsDefaults = function(overrides) {
        const options = {
            filterWidth: 1.0,
            feedback: 0.12,
            lodBias: -1.0,
            sharpness: 0.0,
            enabled: false,
            upscaling: false,
            filterHistory: true,
            filterInput: true,
            useYCoCg: false,
            boxType: Filament.View$TemporalAntiAliasingOptions$BoxType.AABB,
            boxClipping: Filament.View$TemporalAntiAliasingOptions$BoxClipping.ACCURATE,
            jitterPattern: Filament.View$TemporalAntiAliasingOptions$JitterPattern.HALTON_23_X16,
            varianceGamma: 1.0,
            preventFlickering: false,
            historyReprojection: true,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setScreenSpaceReflectionsOptionsDefaults = function(overrides) {
        const options = {
            thickness: 0.1,
            bias: 0.01,
            maxDistance: 3.0,
            stride: 2.0,
            enabled: false,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setGuardBandOptionsDefaults = function(overrides) {
        const options = {
            enabled: false,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setVsmShadowOptionsDefaults = function(overrides) {
        const options = {
            anisotropy: 0,
            mipmapping: false,
            msaaSamples: 1,
            highPrecision: false,
            minVarianceScale: 0.5,
            lightBleedReduction: 0.15,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setSoftShadowOptionsDefaults = function(overrides) {
        const options = {
            penumbraScale: 1.0,
            penumbraRatioScale: 1.0,
        };
        return Object.assign(options, overrides);
    };

    Filament.View.prototype.setStereoscopicOptionsDefaults = function(overrides) {
        const options = {
            enabled: false,
        };
        return Object.assign(options, overrides);
    };

};
