// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;
use std::sync::Arc;

use anyhow::Result;
use common_telemetry::{logging, metric};
use common_test_util::temp_dir::create_temp_dir;
use object_store::layers::LruCacheLayer;
use object_store::services::{Fs, S3};
use object_store::test_util::TempFolder;
use object_store::{util, ObjectStore, ObjectStoreBuilder};
use opendal::raw::Accessor;
use opendal::services::{Azblob, Gcs, Oss};
use opendal::{EntryMode, Operator, OperatorBuilder};

async fn test_object_crud(store: &ObjectStore) -> Result<()> {
    // Create object handler.
    // Write data info object;
    let file_name = "test_file";
    store.write(file_name, "Hello, World!").await?;

    // Read data from object;
    let bs = store.read(file_name).await?;
    assert_eq!("Hello, World!", String::from_utf8(bs)?);

    // Read range from object;
    let bs = store.range_read(file_name, 1..=11).await?;
    assert_eq!("ello, World", String::from_utf8(bs)?);

    // Get object's Metadata
    let meta = store.stat(file_name).await?;
    assert_eq!(EntryMode::FILE, meta.mode());
    assert_eq!(13, meta.content_length());

    // Delete object.
    store.delete(file_name).await.unwrap();
    assert!(store.read(file_name).await.is_err());
    Ok(())
}

async fn test_object_list(store: &ObjectStore) -> Result<()> {
    // Create  some object handlers.
    // Write something
    let p1 = "test_file1";
    let p2 = "test_file2";
    let p3 = "test_file3";
    store.write(p1, "Hello, object1!").await?;
    store.write(p2, "Hello, object2!").await?;
    store.write(p3, "Hello, object3!").await?;

    // List objects
    let lister = store.list("/").await?;
    let entries = util::collect(lister).await?;
    assert_eq!(3, entries.len());

    store.delete(p1).await?;
    store.delete(p3).await?;

    // List objects again
    // Only o2 is exists
    let entries = util::collect(store.list("/").await?).await?;
    assert_eq!(1, entries.len());
    assert_eq!(p2, entries.get(0).unwrap().path());

    let content = store.read(p2).await?;
    assert_eq!("Hello, object2!", String::from_utf8(content)?);

    store.delete(p2).await?;
    let entries = util::collect(store.list("/").await?).await?;
    assert!(entries.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_fs_backend() -> Result<()> {
    let data_dir = create_temp_dir("test_fs_backend");
    let tmp_dir = create_temp_dir("test_fs_backend");
    let mut builder = Fs::default();
    let _ = builder
        .root(&data_dir.path().to_string_lossy())
        .atomic_write_dir(&tmp_dir.path().to_string_lossy());

    let store = ObjectStore::new(builder).unwrap().finish();

    test_object_crud(&store).await?;
    test_object_list(&store).await?;

    Ok(())
}

#[tokio::test]
async fn test_s3_backend() -> Result<()> {
    logging::init_default_ut_logging();
    if let Ok(bucket) = env::var("GT_S3_BUCKET") {
        if !bucket.is_empty() {
            logging::info!("Running s3 test.");

            let root = uuid::Uuid::new_v4().to_string();

            let mut builder = S3::default();
            let _ = builder
                .root(&root)
                .access_key_id(&env::var("GT_S3_ACCESS_KEY_ID")?)
                .secret_access_key(&env::var("GT_S3_ACCESS_KEY")?)
                .region(&env::var("GT_S3_REGION")?)
                .bucket(&bucket);

            let store = ObjectStore::new(builder).unwrap().finish();

            let guard = TempFolder::new(&store, "/");
            test_object_crud(&store).await?;
            test_object_list(&store).await?;
            guard.remove_all().await?;
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_oss_backend() -> Result<()> {
    logging::init_default_ut_logging();
    if let Ok(bucket) = env::var("GT_OSS_BUCKET") {
        if !bucket.is_empty() {
            logging::info!("Running oss test.");

            let root = uuid::Uuid::new_v4().to_string();

            let mut builder = Oss::default();
            let _ = builder
                .root(&root)
                .access_key_id(&env::var("GT_OSS_ACCESS_KEY_ID")?)
                .access_key_secret(&env::var("GT_OSS_ACCESS_KEY")?)
                .bucket(&bucket);

            let store = ObjectStore::new(builder).unwrap().finish();

            let guard = TempFolder::new(&store, "/");
            test_object_crud(&store).await?;
            test_object_list(&store).await?;
            guard.remove_all().await?;
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_azblob_backend() -> Result<()> {
    logging::init_default_ut_logging();
    if let Ok(container) = env::var("GT_AZBLOB_CONTAINER") {
        if !container.is_empty() {
            logging::info!("Running azblob test.");

            let root = uuid::Uuid::new_v4().to_string();

            let mut builder = Azblob::default();
            let _ = builder
                .root(&root)
                .account_name(&env::var("GT_AZBLOB_ACCOUNT_NAME")?)
                .account_key(&env::var("GT_AZBLOB_ACCOUNT_KEY")?)
                .container(&container);

            let store = ObjectStore::new(builder).unwrap().finish();

            let guard = TempFolder::new(&store, "/");
            test_object_crud(&store).await?;
            test_object_list(&store).await?;
            guard.remove_all().await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn test_gcs_backend() -> Result<()> {
    logging::init_default_ut_logging();
    if let Ok(container) = env::var("GT_AZBLOB_CONTAINER") {
        if !container.is_empty() {
            logging::info!("Running azblob test.");

            let mut builder = Gcs::default();
            builder
                .root(&uuid::Uuid::new_v4().to_string())
                .bucket(&env::var("GT_GCS_BUCKET").unwrap())
                .scope(&env::var("GT_GCS_SCOPE").unwrap())
                .credential_path(&env::var("GT_GCS_CREDENTIAL_PATH").unwrap())
                .endpoint(&env::var("GT_GCS_ENDPOINT").unwrap());

            let store = ObjectStore::new(builder).unwrap().finish();

            let guard = TempFolder::new(&store, "/");
            test_object_crud(&store).await?;
            test_object_list(&store).await?;
            guard.remove_all().await?;
        }
    }
    Ok(())
}

async fn assert_lru_cache<C: Accessor + Clone>(
    cache_layer: &LruCacheLayer<C>,
    file_names: &[&str],
) {
    for file_name in file_names {
        assert!(cache_layer.lru_contains_key(file_name).await);
    }
}

async fn assert_cache_files(
    store: &Operator,
    file_names: &[&str],
    file_contents: &[&str],
) -> Result<()> {
    let obs = store.list("/").await?;
    let objects = util::collect(obs).await?;

    // compare the cache file with the expected cache file; ignore orders
    for o in objects {
        let position = file_names.iter().position(|&x| x == o.name());
        assert!(position.is_some(), "file not found: {}", o.name());

        let position = position.unwrap();
        let bs = store.read(o.path()).await.unwrap();
        assert_eq!(
            file_contents[position],
            String::from_utf8(bs.clone())?,
            "file content not match: {}",
            o.name()
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_object_store_cache_policy() -> Result<()> {
    common_telemetry::init_default_ut_logging();
    common_telemetry::init_default_metrics_recorder();
    // create file storage
    let root_dir = create_temp_dir("test_object_store_cache_policy");
    let store = OperatorBuilder::new(
        Fs::default()
            .root(&root_dir.path().to_string_lossy())
            .atomic_write_dir(&root_dir.path().to_string_lossy())
            .build()
            .unwrap(),
    )
    .finish();

    // create file cache layer
    let cache_dir = create_temp_dir("test_object_store_cache_policy_cache");
    let mut builder = Fs::default();
    let _ = builder
        .root(&cache_dir.path().to_string_lossy())
        .atomic_write_dir(&cache_dir.path().to_string_lossy());
    let cache_accessor = Arc::new(builder.build().unwrap());
    let cache_store = OperatorBuilder::new(cache_accessor.clone()).finish();

    // create operator for cache dir to verify cache file
    let cache_layer = LruCacheLayer::new(Arc::new(cache_accessor.clone()), 3)
        .await
        .unwrap();
    let store = store.layer(cache_layer.clone());

    // create several object handler.
    // write data into object;
    let p1 = "test_file1";
    let p2 = "test_file2";
    store.write(p1, "Hello, object1!").await.unwrap();
    store.write(p2, "Hello, object2!").await.unwrap();

    // create cache by read object
    let _ = store.range_read(p1, 0..).await?;
    let _ = store.read(p1).await?;
    let _ = store.range_read(p2, 0..).await?;
    let _ = store.range_read(p2, 7..).await?;
    let _ = store.read(p2).await?;

    assert_cache_files(
        &cache_store,
        &[
            "6d29752bdc6e4d5ba5483b96615d6c48.cache-bytes=0-",
            "ecfe0dce85de452eb0a325158e7bfb75.cache-bytes=7-",
            "ecfe0dce85de452eb0a325158e7bfb75.cache-bytes=0-",
        ],
        &["Hello, object1!", "object2!", "Hello, object2!"],
    )
    .await?;
    assert_lru_cache(
        &cache_layer,
        &[
            "6d29752bdc6e4d5ba5483b96615d6c48.cache-bytes=0-",
            "ecfe0dce85de452eb0a325158e7bfb75.cache-bytes=0-",
        ],
    )
    .await;

    store.delete(p2).await.unwrap();

    assert_cache_files(
        &cache_store,
        &["6d29752bdc6e4d5ba5483b96615d6c48.cache-bytes=0-"],
        &["Hello, object1!"],
    )
    .await?;
    assert_lru_cache(
        &cache_layer,
        &["6d29752bdc6e4d5ba5483b96615d6c48.cache-bytes=0-"],
    )
    .await;

    let p3 = "test_file3";
    store.write(p3, "Hello, object3!").await.unwrap();

    let _ = store.read(p3).await.unwrap();
    let _ = store.range_read(p3, 0..5).await.unwrap();

    assert_cache_files(
        &cache_store,
        &[
            "6d29752bdc6e4d5ba5483b96615d6c48.cache-bytes=0-",
            "a8b1dc21e24bb55974e3e68acc77ed52.cache-bytes=0-",
            "a8b1dc21e24bb55974e3e68acc77ed52.cache-bytes=0-4",
        ],
        &["Hello, object1!", "Hello, object3!", "Hello"],
    )
    .await?;
    assert_lru_cache(
        &cache_layer,
        &[
            "6d29752bdc6e4d5ba5483b96615d6c48.cache-bytes=0-",
            "a8b1dc21e24bb55974e3e68acc77ed52.cache-bytes=0-",
            "a8b1dc21e24bb55974e3e68acc77ed52.cache-bytes=0-4",
        ],
    )
    .await;

    let handle = metric::try_handle().unwrap();
    let metric_text = handle.render();

    assert!(metric_text.contains("object_store_lru_cache_hit"));
    assert!(metric_text.contains("object_store_lru_cache_miss"));

    drop(cache_layer);
    let cache_layer = LruCacheLayer::new(Arc::new(cache_accessor), 3)
        .await
        .unwrap();

    assert_lru_cache(
        &cache_layer,
        &[
            "6d29752bdc6e4d5ba5483b96615d6c48.cache-bytes=0-",
            "a8b1dc21e24bb55974e3e68acc77ed52.cache-bytes=0-",
            "a8b1dc21e24bb55974e3e68acc77ed52.cache-bytes=0-4",
        ],
    )
    .await;

    Ok(())
}