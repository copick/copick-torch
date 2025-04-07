        print(f"Loaded {len(self._subvolumes)} subvolumes with {len(self._keys)} classes")
        print(f"Background samples: {sum(self._is_background)}")
        
    def _sample_background_points(self, tomogram_array, particle_coords):
        """Sample background points away from particles."""
        if not particle_coords:
            return
            
        # Convert to numpy array for distance calculations
        particle_coords = np.array(particle_coords)
        
        # Calculate number of background samples based on ratio
        num_particles = len(particle_coords)
        num_background = int(num_particles * self.background_ratio)
        
        # Limit attempts to avoid infinite loop
        max_attempts = num_background * 10
        attempts = 0
        bg_points_found = 0
        
        half_box = np.array(self.boxsize) // 2
        
        while bg_points_found < num_background and attempts < max_attempts:
            # Generate random point within tomogram bounds with margin for box extraction
            random_point = np.array([
                np.random.randint(half_box[0], tomogram_array.shape[2] - half_box[0]),
                np.random.randint(half_box[1], tomogram_array.shape[1] - half_box[1]),
                np.random.randint(half_box[2], tomogram_array.shape[0] - half_box[2])
            ])
            
            # Calculate distances to all particles
            distances = np.linalg.norm(particle_coords - random_point, axis=1)
            
            # Check if point is far enough from all particles
            if np.min(distances) >= self.min_background_distance:
                # Extract subvolume
                x, y, z = random_point
                subvolume, is_valid, _ = self._extract_subvolume_with_validation(
                    tomogram_array, x, y, z
                )
                
                if is_valid:
                    self._subvolumes.append(subvolume)
                    self._molecule_ids.append(-1)  # Use -1 to indicate background
                    self._is_background.append(True)
                    bg_points_found += 1
            
            attempts += 1
        
        print(f"Added {bg_points_found} background points after {attempts} attempts")
        
    def _augment_subvolume(self, subvolume, idx=None):
        """Apply data augmentation to a subvolume.
        
        This simplified version applies basic augmentations only (no mixup).
        
        Args:
            subvolume: The 3D volume to augment
            idx: Optional index for mixup augmentation (not used in this version)
            
        Returns:
            Augmented subvolume
        """
        # Apply random brightness adjustment
        if random.random() < 0.3:
            delta = np.random.uniform(-0.5, 0.5)
            subvolume = subvolume + delta
            
        # Apply random Gaussian blur
        if random.random() < 0.2:
            sigma = np.random.uniform(0.5, 1.5)
            subvolume = gaussian_filter(subvolume, sigma=sigma)
            
        # Apply random intensity scaling
        if random.random() < 0.2:
            factor = np.random.uniform(0.5, 1.5)
            subvolume = subvolume * factor
            
        # Apply random flip
        if random.random() < 0.2:
            axis = random.randint(0, 2)
            subvolume = np.flip(subvolume, axis=axis)
        
        # Apply random rotation
        if random.random() < 0.2:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            axes = tuple(random.sample([0, 1, 2], 2))  # Select 2 random axes
            subvolume = np.rot90(subvolume, k=k, axes=axes)
            
        return subvolume
    
    def __len__(self):
        """Get the total number of items in the dataset."""
        return len(self._subvolumes)
    
    def get_sample_weights(self):
        """Return sample weights for use in a WeightedRandomSampler."""
        return self.sample_weights
    
    def keys(self):
        """Get pickable object keys."""
        return self._keys
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        class_counts = Counter(self._molecule_ids)
        
        # Create a readable distribution
        distribution = {}
        
        # Count background samples if any
        if -1 in class_counts:
            distribution["background"] = class_counts[-1]
            del class_counts[-1]
        
        # Count regular classes
        for cls_idx, count in class_counts.items():
            if 0 <= cls_idx < len(self._keys):
                distribution[self._keys[cls_idx]] = count
        
        return distribution