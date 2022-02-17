<?php

require "init.php";
$file=fopen('data.txt','r');
while($line=fgets($file))
{
	$var = explode(':', $line, 2);
	$title=$var[0];
	$message=$var[1];
}
$path_to_fcm='http://fcm.googleapis.com/fcm/send';
$server_key="AAAAVWy20VY:APA91bEGlpQkwFsjZQPHozKae8rIewwaRVMYqgnAI_f-rEWb5u-BI_QQs3_SOlqY5AUmBP3xr4fXQfmHjrq5WEv8ge1QJkXNcCz261KIB3fWrKcZsxgPECLMJFr4-xYzS6z0qcnroZQK";
$sql="select fcm_token from fcm_info3";
$result=mysqli_query($con,$sql);
$row=mysqli_fetch_row($result);
$key=$row[0];


$headers=array(
			'Authorization:key=' .$server_key,
			'Content-Type:application/json'
		);
		
$fields = array('to'=>$key,
				'notification'=>array('title'=>$title,'body'=>$message,
				'click_action'=>'com.example.venkatesh.pushnotifi_TARGET_NOTIFICATION'
				));
				
$payload=json_encode($fields);

$curl_session= curl_init();
curl_setopt($curl_session,CURLOPT_URL,$path_to_fcm);
curl_setopt($curl_session,CURLOPT_POST,true);
curl_setopt($curl_session,CURLOPT_HTTPHEADER,$headers);
curl_setopt($curl_session,CURLOPT_RETURNTRANSFER,true);
curl_setopt($curl_session,CURLOPT_SSL_VERIFYPEER,false);
curl_setopt($curl_session,CURLOPT_IPRESOLVE,CURL_IPRESOLVE_V4);
curl_setopt($curl_session,CURLOPT_POSTFIELDS,$payload);
$result = curl_exec($curl_session);
curl_close($curl_session);
mysqli_close($con);		
?>