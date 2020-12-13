var app = angular.module('mainApp', []);
app.controller('mainController', function($scope) {
  
  $scope.showVideoSummarization = function(){
    $scope.videoSummarization = true;
    $scope.implementationDetails = false;
    $scope.results = false;
    $scope.references = false;
  }

  $scope.showImplementationDetails = function(){
    $scope.videoSummarization = false;
    $scope.implementationDetails = true;
    $scope.results = false;
    $scope.references = false;
  }

  $scope.showResults = function(){
    $scope.videoSummarization = false;
    $scope.implementationDetails = false;
    $scope.results = true;
    $scope.references = false;
  }

  $scope.showReferences = function(){
    $scope.videoSummarization = false;
    $scope.implementationDetails = false;
    $scope.results = false;
    $scope.references = true;
  }

  $scope.showOurInfo = function(){
    $scope.ourInfo = true;
  }


  $scope.showProject = function(){
    $scope.project= true;
    $scope.us= false;
    $scope.showVideoSummarization();
  }

  $scope.showUs = function(){
    $scope.project= false;
    $scope.us= true;
    $scope.showOurInfo();
  }

  $scope.showProject();

});