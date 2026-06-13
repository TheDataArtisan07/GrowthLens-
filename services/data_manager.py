import os
import pandas as pd
import numpy as np
import logging

# Configure logger
logger = logging.getLogger('growthlens.data_manager')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class DataManager:
    def __init__(self):
        self.data_folder = None
        self.datasets = {}          # Holds raw DataFrames
        self.dataset_status = {}    # Holds health/summary metrics for each CSV
        self.analytics_df = None    # Cached merged master DataFrame
        self.rfm_df = None          # Cached RFM DataFrame
        self.retention_matrix = None # Cached Cohort Retention Matrix
        self.retention_metrics = None # Cached Cohort Retention Metrics
        self.diagnostics = {
            "shape": (0, 0),
            "unique_customers": 0,
            "unique_orders": 0,
            "unique_products": 0,
            "total_revenue": 0.0,
            "avg_review_score": 0.0,
            "load_success": False
        }
        
        # Define metadata about the datasets to manage
        self.dataset_definitions = {
            "customers": {
                "file": "olist_customers_dataset.csv",
                "required_columns": ['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']
            },
            "orders": {
                "file": "olist_orders_dataset.csv",
                "required_columns": ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                                     'order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date']
            },
            "order_items": {
                "file": "olist_order_items_dataset.csv",
                "required_columns": ['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value']
            },
            "payments": {
                "file": "olist_order_payments_dataset.csv",
                "required_columns": ['order_id', 'payment_sequential', 'payment_type', 'payment_installments', 'payment_value']
            },
            "reviews": {
                "file": "olist_order_reviews_dataset.csv",
                "required_columns": ['order_id', 'review_id', 'review_score']
            },
            "products": {
                "file": "olist_products_dataset.csv",
                "required_columns": ['product_id', 'product_category_name']
            },
            "sellers": {
                "file": "olist_sellers_dataset.csv",
                "required_columns": ['seller_id', 'seller_city', 'seller_state']
            },
            "geolocation": {
                "file": "olist_geolocation_dataset.csv",
                "required_columns": ['geolocation_zip_code_prefix', 'geolocation_city', 'geolocation_state']
            },
            "category_translation": {
                "file": "product_category_name_translation.csv",
                "required_columns": ['product_category_name', 'product_category_name_english']
            }
        }

    def init_app(self, app):
        """
        Initialization hook called during Flask startup.
        Loads, validates, cleans, and merges all CSVs.
        """
        self.data_folder = app.config.get('DATA_FOLDER')
        logger.info(f"Initializing DataManager with data folder: {self.data_folder}")
        
        # 1. Load & Validate
        self.load_all_datasets()
        
        # 2. Clean & Translate
        self.clean_datasets()
        
        # 3. Merge & Create Analytics DataFrame
        self.create_analytics_dataframe()

    def load_all_datasets(self):
        """
        Loads all CSV datasets dynamically. Implements a fail-safe catch
        to log errors and keep the app running even if a load fails.
        """
        for name, info in self.dataset_definitions.items():
            filepath = os.path.join(self.data_folder, info["file"])
            self.dataset_status[name] = {
                "dataset_name": name,
                "file_name": info["file"],
                "status": "Pending",
                "rows": 0,
                "columns": 0,
                "missing_values": 0,
                "duplicates": 0,
                "error_message": ""
            }
            
            if not os.path.exists(filepath):
                msg = f"File not found: {filepath}"
                logger.warning(msg)
                self.dataset_status[name]["status"] = "Failed"
                self.dataset_status[name]["error_message"] = "File missing from disk"
                continue
                
            try:
                # Load the dataset
                df = pd.read_csv(filepath)
                self.datasets[name] = df
                
                # Perform basic validation
                summary = self.validate_dataset_df(df, name, info["required_columns"])
                self.dataset_status[name].update(summary)
                self.dataset_status[name]["status"] = "Loaded"
                logger.info(f"Successfully loaded and validated {name} ({len(df)} rows)")
                
            except Exception as e:
                msg = f"Failed to load dataset {name}: {str(e)}"
                logger.error(msg, exc_info=True)
                self.dataset_status[name]["status"] = "Failed"
                self.dataset_status[name]["error_message"] = str(e)

    def validate_dataset_df(self, df, name, required_cols) -> dict:
        """
        Checks rows, columns, missing values, duplicates, and column presence.
        """
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        rows, cols = df.shape
        missing_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        
        return {
            "rows": rows,
            "columns": cols,
            "missing_values": missing_count,
            "duplicates": duplicate_count
        }

    def clean_datasets(self):
        """
        Cleans loaded datasets: removes duplicates and converts date strings to datetime formats.
        """
        # A. Orders cleaning
        if 'orders' in self.datasets:
            df = self.datasets['orders']
            # Remove duplicate rows
            df = df.drop_duplicates()
            # Convert datetime columns
            date_cols = [
                'order_purchase_timestamp', 
                'order_approved_at', 
                'order_delivered_customer_date', 
                'order_estimated_delivery_date'
            ]
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            self.datasets['orders'] = df

        # B. Clean items/payments/customers/reviews/products/sellers by removing raw duplicates
        for name in ['customers', 'order_items', 'payments', 'reviews', 'products', 'sellers']:
            if name in self.datasets:
                self.datasets[name] = self.datasets[name].drop_duplicates()

        # C. Category Translation
        if 'products' in self.datasets:
            products_df = self.datasets['products']
            if 'category_translation' in self.datasets:
                trans_df = self.datasets['category_translation']
                # Join products with English category translations
                products_df = products_df.merge(
                    trans_df[['product_category_name', 'product_category_name_english']], 
                    on='product_category_name', 
                    how='left'
                )
            else:
                # Fallback: if translation file missing, create the English column from the Portuguese one
                products_df['product_category_name_english'] = products_df['product_category_name']
                
            # Fill missing/NaN translation values with the original Portuguese category name
            if 'product_category_name_english' in products_df.columns:
                products_df['product_category_name_english'] = products_df['product_category_name_english'].fillna(
                    products_df['product_category_name']
                )
                # If still null, fill with 'unknown'
                products_df['product_category_name_english'] = products_df['product_category_name_english'].fillna('unknown')
            else:
                products_df['product_category_name_english'] = 'unknown'
                
            self.datasets['products'] = products_df

    def create_analytics_dataframe(self):
        """
        Merges datasets using LEFT JOINs to preserve data, beginning with 'orders'.
        Calculates derived columns and builds diagnostic figures.
        """
        if 'orders' not in self.datasets:
            logger.error("Cannot create analytics dataframe: 'orders' dataset not loaded.")
            self.diagnostics["load_success"] = False
            return
            
        try:
            # 1. Centering on orders
            merged_df = self.datasets['orders'].copy()
            
            # 2. Join customers
            if 'customers' in self.datasets:
                merged_df = merged_df.merge(self.datasets['customers'], on='customer_id', how='left')
                
            # 3. Join order_items
            if 'order_items' in self.datasets:
                merged_df = merged_df.merge(self.datasets['order_items'], on='order_id', how='left')
                
            # 4. Join payments
            if 'payments' in self.datasets:
                # Olist payments can have multiple records per order. Clean join.
                merged_df = merged_df.merge(self.datasets['payments'], on='order_id', how='left')
                
            # 5. Join reviews
            if 'reviews' in self.datasets:
                # Reviews are merged on order_id. An order might have multiple reviews (rare but possible).
                merged_df = merged_df.merge(self.datasets['reviews'], on='order_id', how='left')
                
            # 6. Join products
            if 'products' in self.datasets:
                merged_df = merged_df.merge(self.datasets['products'], on='product_id', how='left')
                
            # 7. Join sellers
            if 'sellers' in self.datasets:
                merged_df = merged_df.merge(self.datasets['sellers'], on='seller_id', how='left')

            # 8. Create Derived Features
            # Convert purchase timestamp to datetime elements
            purchase_dt = merged_df['order_purchase_timestamp']
            merged_df['purchase_year'] = purchase_dt.dt.year
            merged_df['purchase_month'] = purchase_dt.dt.month
            merged_df['purchase_day'] = purchase_dt.dt.day
            merged_df['purchase_weekday'] = purchase_dt.dt.weekday
            merged_df['purchase_year_month'] = purchase_dt.dt.strftime('%Y-%m')
            
            # Delivery Days = order_delivered_customer_date - order_purchase_timestamp
            if 'order_delivered_customer_date' in merged_df.columns:
                # Parse as timedelta, convert to floating point days
                delivery_delta = merged_df['order_delivered_customer_date'] - purchase_dt
                merged_df['delivery_days'] = delivery_delta.dt.total_seconds() / 86400.0
            else:
                merged_df['delivery_days'] = np.nan
                
            # Review Buckets (Poor: 1-2, Average: 3, Good: 4-5)
            if 'review_score' in merged_df.columns:
                # Vectorized binning for speed
                conditions = [
                    merged_df['review_score'].isin([1.0, 2.0]),
                    merged_df['review_score'] == 3.0,
                    merged_df['review_score'].isin([4.0, 5.0])
                ]
                choices = ['Poor', 'Average', 'Good']
                merged_df['review_bucket'] = np.select(conditions, choices, default='Unknown')
            else:
                merged_df['review_bucket'] = 'Unknown'
                
            # Order value = mapped directly from item price
            if 'price' in merged_df.columns:
                merged_df['order_value'] = merged_df['price']
            else:
                merged_df['order_value'] = np.nan

            # Ensure all required output columns are present (with fallbacks if missing)
            required_output_cols = [
                'customer_id', 'customer_unique_id', 'customer_city', 'customer_state',
                'order_id', 'order_status', 'order_purchase_timestamp',
                'payment_value', 'review_score', 'product_id', 'product_category_name_english',
                'seller_id', 'seller_city', 'seller_state',
                'purchase_year', 'purchase_month', 'purchase_day', 'purchase_weekday', 'purchase_year_month',
                'delivery_days', 'review_bucket', 'order_value',
                'order_delivered_customer_date', 'order_estimated_delivery_date'
            ]
            
            for col in required_output_cols:
                if col not in merged_df.columns:
                    merged_df[col] = np.nan
                    
            # Final cached DataFrame selection
            self.analytics_df = merged_df[required_output_cols]
            
            # Calculate and cache RFM segments
            self.calculate_rfm()
            
            # Calculate and cache Cohort Retention Matrix
            self.calculate_retention_data()
            
            # 9. Compute & Cache Diagnostics
            self.calculate_diagnostics()
            self.diagnostics["load_success"] = True
            
            # Print diagnostic summary in console
            logger.info("=========================================")
            logger.info(f"Master analytics_df Created successfully!")
            logger.info(f"Shape: {self.diagnostics['shape']}")
            logger.info(f"Unique Customers: {self.diagnostics['unique_customers']}")
            logger.info(f"Unique Orders: {self.diagnostics['unique_orders']}")
            logger.info(f"Unique Products: {self.diagnostics['unique_products']}")
            logger.info(f"Total Revenue: ${self.diagnostics['total_revenue']:,.2f}")
            logger.info("=========================================")
            
        except Exception as e:
            logger.error(f"Failed to merge and compile analytics_df: {str(e)}", exc_info=True)
            self.diagnostics["load_success"] = False

    def calculate_diagnostics(self):
        """
        Calculates distinct validation shapes and metrics for the master df.
        """
        df = self.analytics_df
        if df is None:
            return
            
        shape = df.shape
        unique_custs = int(df['customer_unique_id'].dropna().nunique())
        unique_ords = int(df['order_id'].dropna().nunique())
        unique_prods = int(df['product_id'].dropna().nunique())
        
        # Calculate average review score
        avg_rev_score = float(df['review_score'].dropna().mean()) if 'review_score' in df.columns else 0.0
        
        # Sum payments cleanly avoiding duplicates from item multiplication
        # We group by order_id + payment_sequential (or payment parameters) to sum the actual unique payments
        # If order_payments is loaded, we can use it, otherwise fallback to grouped payment value
        total_rev = 0.0
        if 'payments' in self.datasets:
            total_rev = float(self.datasets['payments']['payment_value'].sum())
        else:
            # Grouped fallback
            total_rev = float(df.drop_duplicates(subset=['order_id', 'payment_value'])['payment_value'].sum())
            
        self.diagnostics.update({
            "shape": shape,
            "unique_customers": unique_custs,
            "unique_orders": unique_ords,
            "unique_products": unique_prods,
            "total_revenue": total_rev,
            "avg_review_score": avg_rev_score
        })

    def calculate_rfm(self):
        """
        Calculates and caches the RFM dataframe in memory on startup.
        """
        df = self.analytics_df
        if df is None or df.empty:
            logger.error("Cannot calculate RFM: analytics_df is empty.")
            return

        logger.info("Computing RFM segmentations...")
        
        # 1. Recency: days since latest order relative to dataset max date
        max_date = df['order_purchase_timestamp'].max()
        
        # Latest purchase per customer
        latest_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].max()
        recency = (max_date - latest_purchase).dt.days
        
        # 2. Frequency: number of unique orders per customer
        frequency = df.groupby('customer_unique_id')['order_id'].nunique()
        
        # 3. Monetary: total payment spent per customer
        # We group by order_id to sum payments uniquely first
        order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
        monetary = order_payments.groupby('customer_unique_id')['payment_value'].sum()
        
        # Build RFM DataFrame
        rfm = pd.DataFrame({
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        }).reset_index()
        
        # Fill missing values
        rfm['recency'] = rfm['recency'].fillna(rfm['recency'].max())
        rfm['frequency'] = rfm['frequency'].fillna(1)
        rfm['monetary'] = rfm['monetary'].fillna(0.0)

        # 4. RFM Scoring
        # Recency score (1-5): lower is better
        # Use rank(method='first') to guarantee unique bin edges and equal sizes
        rfm['R_Score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        
        # Frequency score (1-5): custom binning due to Olist's low repeat order counts
        def get_f_score(f):
            if f == 1:
                return 1
            elif f == 2:
                return 3
            elif f == 3:
                return 4
            else:
                return 5
        rfm['F_Score'] = rfm['frequency'].apply(get_f_score)
        
        # Monetary score (1-5): higher is better
        rfm['M_Score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Combine into RFM score string
        rfm['rfm_score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # 5. Customer Segments
        # Let's map to segments based on R and F scores (industry standard)
        def segment_customer(row):
            r = row['R_Score']
            f = row['F_Score']
            
            if r >= 4 and f >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f == 2:
                return 'Potential Loyalists'
            elif r >= 4 and f == 1:
                return 'Recent Customers'
            elif r <= 2 and f >= 3:
                return 'At Risk'
            elif r <= 2 and f == 2:
                return 'Need Attention'
            elif r <= 2 and f == 1:
                return 'Lost Customers'
            else:
                if f == 1:
                    return 'Lost Customers'
                else:
                    return 'Need Attention'

        rfm['segment'] = rfm.apply(segment_customer, axis=1)
        self.rfm_df = rfm
        logger.info(f"RFM computation complete. Cached {len(rfm)} customer segments.")

    # Public API Methods
    def get_analytics_df(self):
        """
        Returns the merged and processed master pandas DataFrame.
        """
        return self.analytics_df

    def get_rfm_df(self):
        """
        Returns the pre-calculated cached RFM segments DataFrame.
        """
        return self.rfm_df

    def get_retention_matrix(self):
        """
        Returns the pre-calculated cached cohort retention percentage matrix.
        """
        return self.retention_matrix

    def get_retention_metrics(self):
        """
        Returns the pre-calculated cached customer lifetime and return interval lists.
        """
        return self.retention_metrics

    def calculate_retention_data(self):
        """
        Calculates and caches the Cohort Retention Matrix and secondary retention metrics in memory on startup.
        """
        df = self.analytics_df
        if df is None or df.empty:
            logger.error("Cannot calculate retention data: analytics_df is empty.")
            return

        logger.info("Computing Cohort Retention Matrix...")
        try:
            # 1. Determine first purchase month per customer
            first_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
            first_purchase.columns = ['customer_unique_id', 'first_purchase_time']
            first_purchase['first_purchase_month'] = first_purchase['first_purchase_time'].dt.strftime('%Y-%m')
            
            # 2. Merge back to main DataFrame
            df_cohort = df.merge(first_purchase, on='customer_unique_id', how='left')
            df_cohort['order_month'] = df_cohort['order_purchase_timestamp'].dt.strftime('%Y-%m')
            
            # Convert string months to Period to easily compute month index delta
            df_cohort['first_purchase_period'] = pd.to_datetime(df_cohort['first_purchase_month']).dt.to_period('M')
            df_cohort['order_period'] = pd.to_datetime(df_cohort['order_month']).dt.to_period('M')
            
            # Calculate months delta
            df_cohort['cohort_index'] = (df_cohort['order_period'] - df_cohort['first_purchase_period']).apply(lambda x: x.n)
            
            # 3. Group by first_purchase_month and cohort_index, count unique customers
            cohort_group = df_cohort.groupby(['first_purchase_month', 'cohort_index'])['customer_unique_id'].nunique().reset_index()
            
            # 4. Pivot to create matrix
            cohort_matrix = cohort_group.pivot(index='first_purchase_month', columns='cohort_index', values='customer_unique_id')
            
            # Save cohort sizes (Column 0 represents Month 0 size)
            cohort_sizes = cohort_matrix.iloc[:, 0].fillna(0).astype(int)
            
            # Normalize to percentages (Month 0 is always 100%)
            retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
            
            # Fill missing column indices (from 0 to max delta) with NaN for alignment
            max_index = int(retention_matrix.columns.max())
            for col in range(max_index + 1):
                if col not in retention_matrix.columns:
                    retention_matrix[col] = np.nan
            retention_matrix = retention_matrix.reindex(columns=sorted(retention_matrix.columns))
            
            self.retention_matrix = retention_matrix
            
            # 5. Compute retention metrics (lifespan, return intervals)
            # A. Lifespan per customer (days between first and latest purchase)
            latest_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
            latest_purchase.columns = ['customer_unique_id', 'latest_purchase_time']
            
            lifespan_df = first_purchase.merge(latest_purchase, on='customer_unique_id')
            lifespan_df['lifespan_days'] = (lifespan_df['latest_purchase_time'] - lifespan_df['first_purchase_time']).dt.days
            
            # B. Return intervals (days between sequential purchases for repeat buyers)
            # Sort orders per customer
            df_sorted = df.drop_duplicates(subset=['order_id']).sort_values(by=['customer_unique_id', 'order_purchase_timestamp'])
            df_sorted['prev_purchase'] = df_sorted.groupby('customer_unique_id')['order_purchase_timestamp'].shift(1)
            df_sorted['interval_days'] = (df_sorted['order_purchase_timestamp'] - df_sorted['prev_purchase']).dt.days
            
            # Cache metrics
            self.retention_metrics = {
                "cohort_sizes": cohort_sizes.to_dict(),
                "lifespans": lifespan_df['lifespan_days'].tolist(),
                "return_intervals": df_sorted['interval_days'].dropna().tolist()
            }
            logger.info("Cohort Retention Matrix calculations complete.")
        except Exception as e:
            logger.error(f"Failed to calculate cohort retention: {str(e)}", exc_info=True)

    def get_dataset_health(self) -> list:
        """
        Returns a summary report for each file's loading status.
        """
        return list(self.dataset_status.values())

    def get_dataset_summary(self) -> dict:
        """
        Returns file definitions status.
        """
        return self.dataset_status

    def get_diagnostics(self) -> dict:
        """
        Returns the diagnostics cache.
        """
        return self.diagnostics

# Instantiate as a singleton service
data_manager = DataManager()
