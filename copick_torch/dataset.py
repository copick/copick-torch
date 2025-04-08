    def _augment_subvolume(self, subvolume, idx=None):
        """Apply data augmentation to a subvolume.
        
        This method uses the AugmentationFactory to create and apply a series of 
        random augmentations to the 3D volume.
        
        Args:
            subvolume: The 3D volume to augment
            idx: Optional index for mixup augmentation (not used in this version)
            
        Returns:
            Augmented subvolume
        """
        # Create a copy of the subvolume to avoid modifying the original
        aug_volume = subvolume.copy()
        
        # Apply random flip - spatial transform
        if random.random() < 0.2:
            axis = random.randint(0, 2)
            aug_volume = np.flip(aug_volume, axis=axis)
        
        # Apply random rotation - spatial transform
        if random.random() < 0.2:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            axes = tuple(random.sample([0, 1, 2], 2))  # Select 2 random axes
            aug_volume = np.rot90(aug_volume, k=k, axes=axes)
        
        # Apply intensity-based augmentations with high probability (80%)
        if random.random() < 0.8:
            # Select 1-3 random augmentation types
            num_augs = random.randint(1, 3)
            
            # Pool of possible augmentations
            all_aug_types = [
                "gaussian_noise",
                "scale_intensity",
                "adjust_contrast",
                "gaussian_smooth",
                "gaussian_sharpen",
                "histogram_shift"
            ]
            
            # Add more complex augmentations with lower probability
            if random.random() < 0.3:
                complex_augs = [
                    "rician_noise", 
                    "gibbs_noise", 
                    "kspace_spike"
                ]
                all_aug_types.extend(complex_augs)
                
            # Randomly select augmentation types
            aug_types = random.sample(all_aug_types, min(num_augs, len(all_aug_types)))
            
            # Create transforms with 100% probability since we already decided to apply them
            transforms = AugmentationFactory.create_transforms(aug_types, prob=1.0)
            
            # Apply transforms - handle both MONAI compose and fallback function
            if hasattr(transforms, '__call__'):
                # Simple function (fallback)
                aug_volume = transforms(aug_volume)
            else:
                # MONAI transform - add and remove channel dimension
                vol_with_channel = aug_volume.copy()[None]  # Add channel dimension
                aug_volume = transforms(vol_with_channel)[0]  # Remove channel dimension
        
        # Apply Fourier domain augmentation with 30% probability
        if random.random() < 0.3:
            fourier_aug = FourierAugment3D(
                freq_mask_prob=0.3,
                phase_noise_std=0.1,
                intensity_scaling_range=(0.8, 1.2)
            )
            aug_volume = fourier_aug(aug_volume)
            
        return aug_volume